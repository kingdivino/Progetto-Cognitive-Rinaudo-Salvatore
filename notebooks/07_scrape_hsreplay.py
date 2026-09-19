"""
Scraper — HSReplay.net (seconda fonte dati REALI per il fine-tuning del classificatore
power level/interestingness, ad affiancare metastats.net).

PERCHE' QUESTO FILE ESISTE (dopo due bocciature precedenti di HSReplay, vedi guida di
progetto): le due verifiche precedenti (scelta iniziale + riconferma 30/08/2026) erano
corrette sui dati che avevano ispezionato, ma non avevano trovato l'endpoint giusto:
1. L'API ufficiale documentata (github.com/HearthSim/hsreplaynet-api-docs) da' accesso
   solo a dati personali dell'account (i propri replay/collezione) - resta vero, non e'
   questo che usiamo qui.
2. La pagina /decks/ era stata ispezionata nel DOM gia' renderizzato, concludendo che
   servisse un browser headless (fragile, rischio scraping aggressivo).
Verifica del 07/09/2026 (ispezione delle richieste di RETE della pagina, non del DOM
finale): la tabella di /decks/ e' popolata da una chiamata XHR a un endpoint JSON
pubblico e NON autenticato:

    GET https://hsreplay.net/analytics/query/list_decks_by_win_rate_v2/
        ?GameType=<...>&LeagueRankRange=<...>&Region=<...>&TimeRange=<...>

Verificato con una richiesta anonima (nessun cookie/login) -> HTTP 200 con JSON diretto.
Niente rendering JS, niente browser headless: un semplice requests.get(), esattamente
come per metastats.net. Schema di ogni mazzo restituito (chiave = classe, es.
"PRIEST", "WARRIOR"...):
    deck_id (str, hash persistente del mazzo - stesso valore ovunque riappaia questo
             mazzo, in qualunque combinazione di filtri)
    deck_list (str, JSON di coppie [dbfId, quantita'], es. "[[111469,1],[121058,2]...]")
    archetype_id (int)
    total_games (int)
    win_rate (float, percentuale)
    avg_game_length_seconds, avg_num_player_turns (extra, non usati qui)

Questa risposta NON include il nome leggibile dell'archetipo (es. "Rafaam Warlock",
"Azalina Priest") - solo archetype_id. Il nome va risolto con una seconda chiamata
(anch'essa pubblica e non autenticata, verificata il 07/09/2026 via fetch() dalla
console del browser su hsreplay.net - una richiesta diretta con requests.get() puo'
essere bloccata da protezioni anti-bot lato server anche se il browser vero passa,
verificarlo al primo run):

    GET https://hsreplay.net/api/v1/archetypes/?hl=en

Ritorna una lista (765 voci circa) di {"id": <archetype_id>, "name": "Rafaam Warlock",
"player_class_name": "WARLOCK", ...} - senza questa risoluzione, domain_data.py cade
sul fallback (solo il nome della classe, es. "WARLOCK") invece del nome
dell'archetipo, come notato dall'utente (07/09/2026) confrontando l'output del
Planner con quello di prima dell'unione con HSReplay.

ATTENZIONE - non tutti i valori dei filtri sono liberi: quelli piu' larghi (rank
"Legend"/"Diamond"/"Platinum"/"ALL", region diversa da "ALL", time range diverso da
quello di patch corrente) sono dietro login/abbonamento Premium e la stessa API
risponde 403 senza autenticazione (verificato con LeagueRankRange=ALL). I valori
liberi, confermati funzionanti in anonimo il 07/09/2026, sono:
    GameType:       RANKED_STANDARD, RANKED_WILD
    LeagueRankRange: BRONZE, SILVER, GOLD, BRONZE_THROUGH_GOLD
    Region:         ALL
    TimeRange:      CURRENT_PATCH
(La UI mostra anche "Last 30 days" e il nome dell'espansione corrente come time range
gratuiti, ma non sono stati verificati qui: possibile estensione futura, vedi TIME_RANGE
sotto - lasciato fisso su CURRENT_PATCH per non rischiare valori indovinati a caso.)

Nota su quanti mazzi UNICI aggiunge davvero: i diversi LeagueRankRange NON sono insiemi
disgiunti (lo stesso mazzo, identificato dallo stesso deck_id, compare tipicamente sia
nella fascia stretta - es. GOLD - sia nell'aggregato BRONZE_THROUGH_GOLD, solo con
total_games diverso). Includerli tutti qui non fa danno (il dedup per deck_id a valle,
in 03_build_finetune_dataset.py, tiene comunque solo la riga con piu' partite) ma non va
contato come moltiplicatore lineare. Il guadagno piu' certo viene da GameType (Standard
e Wild sono pool di mazzi realmente disgiunti) e dal fatto che questa fonte si aggiorna
nell'ordine di ore (vedi "Last updated" sulla pagina), non di settimane come si e'
osservato con metastats.net: rilanciare questo script piu' spesso fa crescere il
dataset in modo molto piu' rapido.

Nessuna decklist da scaricare pagina per pagina (a differenza di metastats.net): la
decklist completa e' gia' dentro la risposta della query - niente cache persistente
per-mazzo, ogni run scrive semplicemente uno snapshot con la marca temporale di oggi,
sullo stesso modello di data/raw/metastats/decks_<data>.csv.

Come eseguirlo (dal venv, con Ollama/Neo4j non necessari qui - e' solo scraping):
    python -u notebooks/07_scrape_hsreplay.py

Output: data/raw/hsreplay/decks_<data>.csv (uno snapshot per esecuzione, stesso spirito
di notebooks/02_scrape_metastats.py - va rilanciato periodicamente, idealmente piu'
spesso che una volta a settimana visto l'aggiornamento frequente della fonte).
"""

import json
import os
import time
from datetime import date

import pandas as pd
import requests

BASE = "https://hsreplay.net"
ANALYTICS_URL = f"{BASE}/analytics/query/list_decks_by_win_rate_v2/"
ARCHETYPES_URL = f"{BASE}/api/v1/archetypes/?hl=en"

HSJSON_DIR = os.path.join("data", "raw", "hearthstonejson")
HSJSON_PATH = os.path.join(HSJSON_DIR, "cards.json")
HSJSON_URL = "https://api.hearthstonejson.com/v1/latest/enUS/cards.json"

OUT_DIR = os.path.join("data", "raw", "hsreplay")

# Solo combinazioni verificate accessibili in anonimo (vedi nota nel docstring sopra).
GAME_TYPES = ["RANKED_STANDARD", "RANKED_WILD"]
RANK_RANGES = ["BRONZE", "SILVER", "GOLD", "BRONZE_THROUGH_GOLD"]
REGION = "ALL"
TIME_RANGE = "CURRENT_PATCH"

REQUEST_DELAY_SECONDS = 1.5  # nessuna decklist da scaricare a parte: poche richieste in totale, delay prudente comunque

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CCAI-BloggerAgent-UniProject/0.1; +https://github.com/)"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def log(msg: str):
    print(msg, flush=True)


def download_hearthstonejson_by_dbfid() -> dict:
    """Scarica/riusa la stessa cache di HearthstoneJSON usata da
    notebooks/03_build_finetune_dataset.py (data/raw/hearthstonejson/cards.json), ma
    indicizzata per dbfId (int) invece che per id stringa: HSReplay identifica le carte
    per dbfId nel campo deck_list, non per l'id stringa (es. "CORE_DS1_185") usato da
    metastats.net. Ritorna dbfId -> id stringa, cosi' l'output di questo script resta
    nello stesso formato (id stringa + quantita') gia' consumato da
    notebooks/03_build_finetune_dataset.py per qualunque fonte."""
    os.makedirs(HSJSON_DIR, exist_ok=True)
    if not os.path.exists(HSJSON_PATH):
        log(f"Scarico HearthstoneJSON da {HSJSON_URL} ...")
        resp = requests.get(HSJSON_URL, timeout=30)
        resp.raise_for_status()
        with open(HSJSON_PATH, "w", encoding="utf-8") as f:
            f.write(resp.text)
        log(f"Salvato in {HSJSON_PATH} ({len(resp.text)} caratteri).")
    else:
        log(f"Uso la cache locale {HSJSON_PATH}.")

    with open(HSJSON_PATH, encoding="utf-8") as f:
        cards = json.load(f)

    by_dbf_id = {}
    for c in cards:
        dbf_id = c.get("dbfId")
        cid = c.get("id")
        if dbf_id is not None and cid:
            by_dbf_id[dbf_id] = cid
    log(f"Carte indicizzate per dbfId: {len(by_dbf_id)}")
    return by_dbf_id


def fetch_archetype_names() -> dict:
    """Risolve archetype_id -> nome leggibile (es. 832 -> "Rafaam Warlock") interrogando
    l'endpoint pubblico degli archetipi. Se la richiesta fallisce (rete, blocco
    anti-bot, ecc.) ritorna un dizionario vuoto invece di far fallire tutto lo script -
    il chiamante deve degradare al nome della classe, non crashare (stesso principio
    gia' usato altrove nel progetto per le fonti esterne non raggiungibili)."""
    try:
        resp = SESSION.get(ARCHETYPES_URL, timeout=20)
        resp.raise_for_status()
        archetypes = resp.json()
        names = {a["id"]: a["name"] for a in archetypes if a.get("id") is not None and a.get("name")}
        log(f"Nomi archetipo risolti: {len(names)}")
        return names
    except Exception as e:
        log(f"  [WARN] Impossibile risolvere i nomi degli archetipi ({e}) - i mazzi HSReplay avranno solo il nome della classe.")
        return {}


def fetch_combo(game_type: str, rank_range: str) -> dict:
    params = {
        "GameType": game_type,
        "LeagueRankRange": rank_range,
        "Region": REGION,
        "TimeRange": TIME_RANGE,
    }
    resp = SESSION.get(ANALYTICS_URL, params=params, timeout=20)
    if resp.status_code == 403:
        log(f"  [SKIP] {game_type}/{rank_range}: 403 (combinazione dietro login/Premium, non dovrebbe capitare con i valori liberi elencati sopra)")
        return {}
    resp.raise_for_status()
    return resp.json()


def parse_combo(payload: dict, game_type: str, rank_range: str, dbf_to_id: dict, archetype_names: dict) -> list:
    rows = []
    data_by_class = (payload or {}).get("series", {}).get("data", {})
    for deck_class, decks in data_by_class.items():
        for d in decks:
            archetype_id = d.get("archetype_id")
            # Fallback al nome della classe se l'archetipo non e' risolvibile (id
            # negativo/non classificato, o fetch_archetype_names() fallito) - non
            # peggio di quanto avveniva prima di questo fix.
            deck_name_raw = archetype_names.get(archetype_id, deck_class)
            deck_list_raw = d.get("deck_list")
            try:
                pairs = json.loads(deck_list_raw) if deck_list_raw else []
            except (ValueError, TypeError):
                pairs = []

            cards_id_count = []
            for entry in pairs:
                if not isinstance(entry, list) or len(entry) != 2:
                    continue
                dbf_id, qty = entry
                card_id = dbf_to_id.get(dbf_id)
                if card_id is None:
                    continue  # dbfId non trovato nel dump (raro: carta nuovissima, cache locale da riscaricare)
                cards_id_count.append((card_id, qty))

            rows.append({
                "deck_id": d.get("deck_id"),
                "deck_name_raw": deck_name_raw,
                "deck_class_hsreplay": deck_class,
                "archetype_id": archetype_id,
                "cards_id_count": cards_id_count,
                "winrate": d.get("win_rate"),
                "games": d.get("total_games"),
                "game_type": game_type,
                "rank_range": rank_range,
                "scrape_date": date.today().isoformat(),
            })
    return rows


def main():
    dbf_to_id = download_hearthstonejson_by_dbfid()
    archetype_names = fetch_archetype_names()

    all_rows = []
    combos = [(gt, rr) for gt in GAME_TYPES for rr in RANK_RANGES]
    for idx, (game_type, rank_range) in enumerate(combos, start=1):
        log(f"\n[{idx}/{len(combos)}] Scarico {game_type} / {rank_range} (region={REGION}, time_range={TIME_RANGE}) ...")
        try:
            payload = fetch_combo(game_type, rank_range)
            rows = parse_combo(payload, game_type, rank_range, dbf_to_id, archetype_names)
            log(f"  [OK] {len(rows)} mazzi.")
            all_rows.extend(rows)
        except Exception as e:
            log(f"  [ERROR] {game_type}/{rank_range}: {e}")
        time.sleep(REQUEST_DELAY_SECONDS)

    if not all_rows:
        log("\n[VERDETTO] Nessun mazzo scaricato - controllare a mano l'endpoint (potrebbe essere cambiato, vedi docstring del modulo per come e' stato scoperto).")
        return

    out_df = pd.DataFrame(all_rows)
    log(f"\nRighe totali (con duplicati: stesso deck_id puo' comparire in piu' combinazioni di filtri): {len(out_df)}")

    # Stesso principio del dedup in 03_build_finetune_dataset.py: tra le combinazioni
    # in cui ricompare lo stesso mazzo, tiene quella con piu' partite (winrate piu'
    # affidabile) - qui e' solo per non salvare righe ovviamente ridondanti nello
    # snapshot di oggi, il dedup "vero" tra fonti diverse resta comunque in 03.
    out_df = out_df.sort_values("games", ascending=False).drop_duplicates(subset="deck_id", keep="first")
    log(f"Mazzi unici in questo snapshot (dopo dedup tra le combinazioni di oggi): {len(out_df)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"decks_{date.today().isoformat()}.csv")
    out_df.to_csv(out_path, index=False)
    log(f"\n[OK] Salvato snapshot di oggi: {len(out_df)} mazzi in {out_path}")

    log("\n--- Statistiche rapide ---")
    log(f"Winrate: min={out_df['winrate'].min():.1f}%, media={out_df['winrate'].mean():.1f}%, max={out_df['winrate'].max():.1f}%")
    log(f"Partite: min={out_df['games'].min():.0f}, mediana={out_df['games'].median():.0f}, max={out_df['games'].max():.0f}")
    log("\nMazzi per classe (etichetta HSReplay, ricalcolata comunque dalle carte in 03_build_finetune_dataset.py):")
    log(out_df["deck_class_hsreplay"].value_counts().to_string())
    log("\nMazzi per game_type:")
    log(out_df["game_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
