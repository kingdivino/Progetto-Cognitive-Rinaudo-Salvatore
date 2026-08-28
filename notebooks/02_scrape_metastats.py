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
  Questo è normale e anzi preferibile alle specifiche del progetto (dati reali > sintetici).

Perché BeautifulSoup e non solo pandas.read_html:
il testo visibile nella colonna "Deck" (es. "Dragon Warrior #12321") contiene
probabilmente caratteri invisibili (zero-width) intercalati tra le cifre — invisibili a
occhio ma che rompono qualunque regex su cifre consecutive (verificato: il pattern
funziona su una stringa scritta a mano, ma fallisce sempre sul testo reale del sito).
Per questo l'id del mazzo si legge dal link href della riga (es.
"/hearthstone/deck/12321/"), che deve essere corretto perché è quello che porta
davvero alla pagina del mazzo — non è soggetto allo stesso trucco del testo visibile.

Nota importante sul volume dati: l'espansione corrente è uscita da pochi giorni,
quindi al momento ci sono pochi mazzi tracciati (~67 mazzi, ~22 archetipi visti finora).
Il piano è rilanciare questo script periodicamente (es. una volta a settimana) per
accumulare dati nel tempo. Ogni run salva un file con la data, senza sovrascrivere i
precedenti.

Come eseguirlo:
    python notebooks/02_scrape_metastats.py

ATTENZIONE:
- Il sito applica rate limiting (abbiamo ricevuto 429 durante l'esplorazione) — questo
  script mette una pausa tra le richieste, NON aumentare la frequenza senza motivo.
"""

import os
import re
import time
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://metastats.net"
WINRATE_URL = f"{BASE}/hearthstone/decks/winrate/"
DECK_URL_TMPL = f"{BASE}/hearthstone/deck/{{deck_id}}/"

OUT_DIR = os.path.join("data", "raw", "metastats")
REQUEST_DELAY_SECONDS = 3  # gentile col sito, non abbassare senza motivo

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CCAI-BloggerAgent-UniProject/0.1; +https://github.com/)"
}

DECK_LINK_RE = re.compile(r"/hearthstone/deck/(\d+)/?")
DECKSTRING_RE = re.compile(r"AAEC[A-Za-z0-9+/=]{20,}")


def fetch_page_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def parse_deck_table(html: str) -> pd.DataFrame:
    """Legge #deck-win-rates riga per riga con BeautifulSoup: id mazzo dal link href
    (robusto), il resto delle colonne dal testo delle celle."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find(id="deck-win-rates")
    if table is None:
        print("  [WARN] Tabella #deck-win-rates non trovata nell'HTML.")
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
    print(f"  [OK] Estratti {len(df)} mazzi da #deck-win-rates (via link href).")
    return df


def fetch_deck_code(deck_id: int) -> str | None:
    html = fetch_page_html(DECK_URL_TMPL.format(deck_id=deck_id))
    match = DECKSTRING_RE.search(html)
    if not match:
        print(f"  [WARN] Nessun deckstring trovato per deck {deck_id}.")
        return None
    return match.group(0)


def decode_deck_code(deckstring: str):
    from hearthstone import deckstrings  # import qui: dipendenza opzionale, serve solo qui
    # Restituisce 4 valori, non 3 come indicava la doc riassunta inizialmente:
    # cards, heroes, format_type, sideboards
    cards, heroes, format_type, sideboards = deckstrings.parse_deckstring(deckstring)
    return cards, heroes, format_type, sideboards  # cards: lista di (dbfId, count)


def main():
    print(f"Scarico {WINRATE_URL} ...")
    html = fetch_page_html(WINRATE_URL)
    print(f"Ricevuti {len(html)} caratteri di HTML.")

    deck_df = parse_deck_table(html)
    if deck_df.empty:
        print("\n[VERDETTO] Nessun mazzo estratto — l'id/href della tabella potrebbe")
        print("avere un formato diverso da quanto previsto. Serve ispezionare l'HTML a mano.")
        return

    print("\nPrime righe estratte:")
    print(deck_df.head())

    rows = []
    for _, deck_row in deck_df.iterrows():
        deck_id = int(deck_row["deck_id"])
        print(f"\nScarico mazzo #{deck_id} ({deck_row['deck_name_raw']}) ...")
        try:
            deckstring = fetch_deck_code(deck_id)
            cards = heroes = format_type = sideboards = None
            if deckstring:
                cards, heroes, format_type, sideboards = decode_deck_code(deckstring)
            rows.append({
                "deck_id": deck_id,
                "deck_name_raw": deck_row["deck_name_raw"],
                "cells": deck_row["cells"],
                "deckstring": deckstring,
                "cards_dbfid_count": cards,
                "heroes": heroes,
                "format": format_type,
                "sideboards": sideboards,
                "scrape_date": date.today().isoformat(),
            })
        except Exception as e:
            print(f"  [ERROR] deck {deck_id}: {e}")
        time.sleep(REQUEST_DELAY_SECONDS)

    out_df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"decks_{date.today().isoformat()}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[OK] Salvato {len(out_df)} mazzi in {out_path}")


if __name__ == "__main__":
    main()
