"""
Scraper — metastats.net (fonte dati REALI e ATTUALI per il fine-tuning del
classificatore power level/interestingness).

Perché questo script esiste:
- HSDataset (Zenodo) copre 2013-2020, troppo vecchio (power creep, archetipi non più
  esistenti) — vedi notebooks/01_check_hsdataset_schema.py e la guida di progetto.
- metastats.net pubblica winrate REALI e ATTUALI per archetipo/mazzo, in tabelle HTML
  semplici, senza bisogno di JavaScript reso lato client (da verificare al primo run).
- NON esiste un dataset scaricabile pronto: va costruito noi stessi con uno scraper.
  Questo è normale e anzi preferibile alle specifiche del progetto (dati reali > sintetici).

Nota importante sul volume dati: l'espansione corrente è uscita da pochi giorni,
quindi al momento ci sono pochi mazzi tracciati (~20 archetipi, ~55 mazzi). Il piano è
rilanciare questo script periodicamente (es. una volta a settimana) per accumulare dati
nel tempo, così quando si arriverà alla fase di fine-tuning (più avanti nella roadmap)
il dataset sarà cresciuto. Ogni run salva un file con la data, senza sovrascrivere i
precedenti.

Come eseguirlo:
    python notebooks/02_scrape_metastats.py

ATTENZIONE:
- Il sito applica rate limiting (abbiamo ricevuto 429 durante l'esplorazione) — questo
  script mette una pausa tra le richieste, NON aumentare la frequenza senza motivo.
- Se lo script non trova nessuna tabella con pandas.read_html, il sito probabilmente
  renderizza la tabella via JavaScript (Single Page App): in quel caso serve un
  approccio diverso (Selenium/Playwright) — segnalarlo e non insistere con richieste
  ripetute.
"""

import os
import re
import time
from datetime import date

import pandas as pd
import requests

BASE = "https://metastats.net"
WINRATE_URL = f"{BASE}/hearthstone/decks/winrate/"
DECK_URL_TMPL = f"{BASE}/hearthstone/deck/{{deck_id}}/"

OUT_DIR = os.path.join("data", "raw", "metastats")
REQUEST_DELAY_SECONDS = 3  # gentile col sito, non abbassare senza motivo

HEADERS = {
    # User-Agent onesto: siamo un progetto universitario, non finge di essere un browser a caso
    "User-Agent": "Mozilla/5.0 (compatible; CCAI-BloggerAgent-UniProject/0.1; +https://github.com/)"
}

DECKSTRING_RE = re.compile(r"AAEC[A-Za-z0-9+/=]{20,}")


def fetch_winrate_tables():
    resp = requests.get(WINRATE_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    print(f"[fetch_winrate_tables] Trovate {len(tables)} tabelle HTML sulla pagina.")
    for i, t in enumerate(tables):
        print(f"\n--- Tabella {i} (shape {t.shape}) ---")
        print(t.head())
    return tables


def extract_deck_ids(tables):
    """Cerca in tutte le tabelle una colonna che contenga pattern tipo 'Nome #12345'
    ed estrae gli id numerici dei mazzi."""
    deck_ids = set()
    pattern = re.compile(r"#(\d+)")
    for t in tables:
        for col in t.columns:
            if t[col].dtype == object:
                for val in t[col].dropna().astype(str):
                    m = pattern.search(val)
                    if m:
                        deck_ids.add(int(m.group(1)))
    print(f"\n[extract_deck_ids] Trovati {len(deck_ids)} id di mazzo unici: {sorted(deck_ids)[:10]}...")
    return sorted(deck_ids)


def fetch_deck_code(deck_id: int) -> str | None:
    url = DECK_URL_TMPL.format(deck_id=deck_id)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    match = DECKSTRING_RE.search(resp.text)
    if not match:
        print(f"  [WARN] Nessun deckstring trovato per deck {deck_id} — pagina forse JS-rendered.")
        return None
    return match.group(0)


def decode_deck_code(deckstring: str):
    from hearthstone import deckstrings  # import qui: dipendenza opzionale, serve solo qui
    cards, heroes, format_type = deckstrings.parse_deckstring(deckstring)
    return cards, heroes, format_type  # cards: lista di (dbfId, count)


def main():
    tables = fetch_winrate_tables()
    if not tables:
        print("\n[VERDETTO] Nessuna tabella trovata da pandas.read_html.")
        print("Probabile causa: la pagina renderizza i dati via JavaScript lato client.")
        print("Non insistere con altre richieste — serve rivedere l'approccio (Selenium/Playwright).")
        return

    deck_ids = extract_deck_ids(tables)
    if not deck_ids:
        print("\n[VERDETTO] Tabelle trovate ma nessun id mazzo estratto — il formato del testo")
        print("nelle celle (es. 'Nome #12345') potrebbe essere diverso da quanto previsto.")
        return

    rows = []
    for deck_id in deck_ids:
        print(f"\nScarico mazzo #{deck_id} ...")
        try:
            deckstring = fetch_deck_code(deck_id)
            cards = heroes = format_type = None
            if deckstring:
                cards, heroes, format_type = decode_deck_code(deckstring)
            rows.append({
                "deck_id": deck_id,
                "deckstring": deckstring,
                "cards_dbfid_count": cards,
                "heroes": heroes,
                "format": format_type,
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
