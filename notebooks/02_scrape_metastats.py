"""
Scraper — metastats.net (fonte dati REALI e ATTUALI per il fine-tuning del
classificatore power level/interestingness).

Perché questo script esiste:
- HSDataset (Zenodo) copre 2013-2020, troppo vecchio (power creep, archetipi non più
  esistenti) — vedi notebooks/01_check_hsdataset_schema.py e la guida di progetto.
- metastats.net pubblica winrate REALI e ATTUALI per archetipo/mazzo. Verificato che è
  HTML renderizzato dal server (jQuery DataTables abbellisce tabelle già presenti nel
  markup, non le carica via JS) — le due tabelle hanno id fissi:
    #deck-win-rates      -> mazzi individuali (nome, id, partite totali, winrate)
    #archetype-win-rates -> aggregato per archetipo (nome, winrate)
- NON esiste un dataset scaricabile pronto: va costruito noi stessi con uno scraper.
  Questo è normale e anzi preferibile alle specifiche del progetto (dati reali > sintetici).

Nota importante sul volume dati: l'espansione corrente è uscita da pochi giorni,
quindi al momento ci sono pochi mazzi tracciati. Il piano è rilanciare questo script
periodicamente (es. una volta a settimana) per accumulare dati nel tempo. Ogni run
salva un file con la data, senza sovrascrivere i precedenti.

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

BASE = "https://metastats.net"
WINRATE_URL = f"{BASE}/hearthstone/decks/winrate/"
DECK_URL_TMPL = f"{BASE}/hearthstone/deck/{{deck_id}}/"

OUT_DIR = os.path.join("data", "raw", "metastats")
REQUEST_DELAY_SECONDS = 3  # gentile col sito, non abbassare senza motivo

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CCAI-BloggerAgent-UniProject/0.1; +https://github.com/)"
}

DECKSTRING_RE = re.compile(r"AAEC[A-Za-z0-9+/=]{20,}")


def fetch_page_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def fetch_table_by_id(html: str, table_id: str) -> pd.DataFrame | None:
    """Estrae UNA tabella specifica per id, invece di prendere tutte le tabelle
    della pagina (più robusto: evita di raccogliere tabelle di layout/pubblicità
    che non c'entrano nulla)."""
    try:
        tables = pd.read_html(html, attrs={"id": table_id})
    except ValueError as e:
        print(f"  [WARN] Nessuna tabella trovata con id='{table_id}': {e}")
        return None
    if not tables:
        return None
    df = tables[0]
    print(f"  [OK] Tabella #{table_id}: shape={df.shape}, colonne={list(df.columns)}")
    return df


def extract_deck_ids(df: pd.DataFrame) -> list[int]:
    """Cerca in tutte le colonne testuali pattern tipo 'Nome #12345' ed estrae gli id."""
    deck_ids = set()
    pattern = re.compile(r"#(\d+)")
    for col in df.columns:
        if df[col].dtype == object:
            for val in df[col].dropna().astype(str):
                m = pattern.search(val)
                if m:
                    deck_ids.add(int(m.group(1)))
    return sorted(deck_ids)


def fetch_deck_code(deck_id: int) -> str | None:
    html = fetch_page_html(DECK_URL_TMPL.format(deck_id=deck_id))
    match = DECKSTRING_RE.search(html)
    if not match:
        print(f"  [WARN] Nessun deckstring trovato per deck {deck_id}.")
        return None
    return match.group(0)


def decode_deck_code(deckstring: str):
    from hearthstone import deckstrings  # import qui: dipendenza opzionale, serve solo qui
    cards, heroes, format_type = deckstrings.parse_deckstring(deckstring)
    return cards, heroes, format_type  # cards: lista di (dbfId, count)


def main():
    print(f"Scarico {WINRATE_URL} ...")
    html = fetch_page_html(WINRATE_URL)
    print(f"Ricevuti {len(html)} caratteri di HTML.")

    deck_df = fetch_table_by_id(html, "deck-win-rates")
    archetype_df = fetch_table_by_id(html, "archetype-win-rates")

    if deck_df is None:
        print("\n[VERDETTO] Tabella #deck-win-rates non trovata da pandas.read_html.")
        print("Possibili cause: lxml non installato (serve 'pip install lxml'), oppure")
        print("l'id della tabella è cambiato sul sito rispetto a quanto verificato.")
        return

    if archetype_df is not None:
        os.makedirs(OUT_DIR, exist_ok=True)
        archetype_path = os.path.join(OUT_DIR, f"archetype_winrates_{date.today().isoformat()}.csv")
        archetype_df.to_csv(archetype_path, index=False)
        print(f"[OK] Salvata tabella archetipi in {archetype_path}")

    deck_ids = extract_deck_ids(deck_df)
    print(f"\n[extract_deck_ids] Trovati {len(deck_ids)} id di mazzo unici: {deck_ids}")
    if not deck_ids:
        print("\n[VERDETTO] Tabella trovata ma nessun id mazzo estratto — il formato del testo")
        print("nelle celle (es. 'Nome #12345') potrebbe essere diverso da quanto previsto.")
        print("Prima riga della tabella per debug:")
        print(deck_df.head(1).to_dict())
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
