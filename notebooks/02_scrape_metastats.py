"""
Scraper — metastats.net (fonte dati REALI e ATTUALI per il fine-tuning del
classificatore power level/interestingness).

Perché questo script esiste:
- HSDataset (Zenodo) copre 2013-2020, troppo vecchio (power creep, archetipi non più
  esistenti) — vedi notebooks/01_check_hsdataset_schema.py e la guida di progetto.
- metastats.net pubblica winrate REALI e ATTUALI per archetipo/mazzo. È HTML
  renderizzato dal server (jQuery DataTables abbellisce tabelle già presenti nel
  markup) con id fissi: #deck-win-rates, #archetype-win-rates.
- NON esiste un dataset scaricabile pronto: va costruito noi stessi con uno scraper.

Perché BeautifulSoup e non solo pandas.read_html per gli id:
il testo visibile nella colonna "Deck" contiene caratteri invisibili (zero-width)
intercalati tra le cifre — invisibili a occhio ma che rompono qualunque regex su cifre
consecutive. Per questo l'id del mazzo si legge dal link href della riga, non dal testo.

Cache persistente (data/raw/metastats/deck_cache.csv):
la DECKLIST di un mazzo con un certo id non cambia mai nel tempo (cambia solo quante
partite/winrate ha nella tabella principale, che viene sempre riletta fresca). Quindi
ai run successivi si riscaricano solo i mazzi MAI visti prima — enormemente più veloce
alla seconda esecuzione in poi, senza perdere nessun dato.

Come eseguirlo (il flag -u disabilita il buffering di Python, utile quando l'output
va su file/pipe invece che a schermo — altrimenti non si vede nulla finché non finisce):
    python -u notebooks/02_scrape_metastats.py

ATTENZIONE:
- Il sito applica rate limiting (visto un 429 durante l'esplorazione manuale). Il delay
  di default è prudente; se il sito inizia a rispondere 429 lo script rallenta da solo
  (backoff) invece di continuare a martellarlo.
"""

import csv
import io
import os
import re
import sys
import time
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://metastats.net"
WINRATE_URL = f"{BASE}/hearthstone/decks/winrate/"
DECK_URL_TMPL = f"{BASE}/hearthstone/deck/{{deck_id}}/"

OUT_DIR = os.path.join("data", "raw", "metastats")
CACHE_PATH = os.path.join(OUT_DIR, "deck_cache.csv")
REQUEST_DELAY_SECONDS = 2  # ridotto da 3 a 2: nessun 429 osservato durante lo scraping vero
MAX_RETRIES_ON_429 = 4

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CCAI-BloggerAgent-UniProject/0.1; +https://github.com/)"
}

DECK_LINK_RE = re.compile(r"/hearthstone/deck/(\d+)/?")
DECKSTRING_RE = re.compile(r"AAEC[A-Za-z0-9+/=]{20,}")

SESSION = requests.Session()  # riusa la connessione TCP/TLS tra una richiesta e l'altra (più veloce, zero rischio in più per il sito)
SESSION.headers.update(HEADERS)


def log(msg: str):
    print(msg, flush=True)  # flush esplicito: senza, l'output resta bloccato nel buffer quando non c'è un terminale


def fetch_page_html(url: str) -> str:
    """GET con backoff automatico se il sito risponde 429 (troppo rate limit)."""
    delay = REQUEST_DELAY_SECONDS
    for attempt in range(1, MAX_RETRIES_ON_429 + 1):
        resp = SESSION.get(url, timeout=15)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", delay * attempt * 2))
            log(f"    [429] Rate limited, aspetto {wait}s (tentativo {attempt}/{MAX_RETRIES_ON_429})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.text
    raise RuntimeError(f"Troppi 429 consecutivi su {url}, mi fermo per non stressare il sito.")


def parse_deck_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find(id="deck-win-rates")
    if table is None:
        log("  [WARN] Tabella #deck-win-rates non trovata nell'HTML.")
        return pd.DataFrame()

    rows = []
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        link = tr.find("a", href=DECK_LINK_RE)
        if not link:
            continue
        m = DECK_LINK_RE.search(link["href"])
        deck_id = int(m.group(1))
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        rows.append({
            "deck_id": deck_id,
            "deck_name_raw": link.get_text(strip=True),
            "cells": cells,
        })
    df = pd.DataFrame(rows)
    log(f"  [OK] Estratti {len(df)} mazzi da #deck-win-rates (via link href).")
    return df


def load_cache() -> dict:
    """Carica la cache deck_id -> (deckstring, cards, heroes, format, sideboards) dai run precedenti."""
    if not os.path.exists(CACHE_PATH):
        return {}
    cache = {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cache[int(row["deck_id"])] = row
    return cache


def save_cache(cache_rows: list):
    os.makedirs(OUT_DIR, exist_ok=True)
    fieldnames = ["deck_id", "deckstring", "cards_dbfid_count", "heroes", "format", "sideboards", "first_seen"]
    with open(CACHE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in cache_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def fetch_deck_code(deck_id: int) -> str | None:
    html = fetch_page_html(DECK_URL_TMPL.format(deck_id=deck_id))
    match = DECKSTRING_RE.search(html)
    if not match:
        log(f"  [WARN] Nessun deckstring trovato per deck {deck_id}.")
        return None
    return match.group(0)


def decode_deck_code(deckstring: str):
    from hearthstone import deckstrings  # import qui: dipendenza opzionale, serve solo qui
    # Restituisce 4 valori: cards, heroes, format_type, sideboards
    return deckstrings.parse_deckstring(deckstring)


def main():
    log(f"Scarico {WINRATE_URL} ...")
    html = fetch_page_html(WINRATE_URL)
    log(f"Ricevuti {len(html)} caratteri di HTML.")

    deck_df = parse_deck_table(html)
    if deck_df.empty:
        log("\n[VERDETTO] Nessun mazzo estratto — l'id/href della tabella potrebbe")
        log("avere un formato diverso da quanto previsto. Serve ispezionare l'HTML a mano.")
        return

    cache = load_cache()
    all_ids = [int(x) for x in deck_df["deck_id"]]

    def has_deckstring(entry: dict) -> bool:
        return bool((entry.get("deckstring") or "").strip())

    never_seen_ids = [i for i in all_ids if i not in cache]
    # I mazzi già visti ma senza deck code trovato l'ultima volta vengono ritentati ad
    # ogni esecuzione (non è detto sia un limite permanente di quella pagina) — solo
    # quelli con decklist già decodificata vengono saltati per davvero.
    retry_ids = [i for i in all_ids if i in cache and not has_deckstring(cache[i])]
    new_ids = never_seen_ids + retry_ids
    cached_ids = [i for i in all_ids if i in cache and has_deckstring(cache[i])]

    est_minutes = len(new_ids) * REQUEST_DELAY_SECONDS / 60
    log(f"\nMazzi totali trovati: {len(all_ids)}")
    log(f"Già in cache con decklist valida (skip): {len(cached_ids)}")
    log(f"Mai visti prima: {len(never_seen_ids)}")
    log(f"Da ritentare (in cache ma senza deck code l'ultima volta): {len(retry_ids)}")
    log(f"Da scaricare adesso in totale: {len(new_ids)} (~{est_minutes:.1f} minuti stimati a {REQUEST_DELAY_SECONDS}s/richiesta)")

    name_by_id = dict(zip(deck_df["deck_id"], deck_df["deck_name_raw"]))
    new_cache_rows = []
    for idx, deck_id in enumerate(new_ids, start=1):
        log(f"\n[{idx}/{len(new_ids)}] Scarico mazzo #{deck_id} ({name_by_id.get(deck_id, '?')}) ...")
        try:
            deckstring = fetch_deck_code(deck_id)
            cards = heroes = format_type = sideboards = None
            if deckstring:
                cards, heroes, format_type, sideboards = decode_deck_code(deckstring)
            new_cache_rows.append({
                "deck_id": deck_id,
                "deckstring": deckstring,
                "cards_dbfid_count": cards,
                "heroes": heroes,
                "format": format_type,
                "sideboards": sideboards,
                "first_seen": date.today().isoformat(),
            })
        except Exception as e:
            log(f"  [ERROR] deck {deck_id}: {e}")
        time.sleep(REQUEST_DELAY_SECONDS)

    # Unisce cache vecchia + nuova: le entry ritentate SOSTITUISCONO quelle vecchie
    # (stesso deck_id), non si accumulano — altrimenti la cache duplicherebbe righe
    # ad ogni ritento.
    cache_by_id = {int(k): v for k, v in cache.items()}
    for row in new_cache_rows:
        cache_by_id[int(row["deck_id"])] = row
    full_cache = list(cache_by_id.values())
    save_cache(full_cache)
    log(f"\n[OK] Cache aggiornata: {len(full_cache)} mazzi totali in {CACHE_PATH}")

    # Costruisce l'output di QUESTO run: tutti i mazzi visti oggi (vecchi+nuovi) con
    # le loro carte (dalla cache) + il winrate/partite di OGGI (dalla tabella appena letta)
    cache_by_id = {int(r["deck_id"]): r for r in full_cache}
    out_rows = []
    for _, deck_row in deck_df.iterrows():
        deck_id = int(deck_row["deck_id"])
        cached = cache_by_id.get(deck_id, {})
        out_rows.append({
            "deck_id": deck_id,
            "deck_name_raw": deck_row["deck_name_raw"],
            "cells": deck_row["cells"],
            "deckstring": cached.get("deckstring"),
            "cards_dbfid_count": cached.get("cards_dbfid_count"),
            "heroes": cached.get("heroes"),
            "format": cached.get("format"),
            "sideboards": cached.get("sideboards"),
            "scrape_date": date.today().isoformat(),
        })

    out_df = pd.DataFrame(out_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"decks_{date.today().isoformat()}.csv")
    out_df.to_csv(out_path, index=False)
    log(f"[OK] Salvato snapshot di oggi: {len(out_df)} mazzi in {out_path}")


if __name__ == "__main__":
    main()
