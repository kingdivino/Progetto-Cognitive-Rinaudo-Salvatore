"""
Regole sul formato di gioco (Standard vs Wild) - aggiunto il 07/09/2026 su
segnalazione dell'utente: in Wild sono legali TUTTE le carte mai pubblicate, in
Standard solo le espansioni attualmente in rotazione (piu' i set permanenti
Basic/Core) - un sottoinsieme. Rischio concreto senza questo modulo: un post su un
mazzo Standard che suggerisce (via RAG, che copre TUTTE le carte di HearthstoneJSON
senza distinzione di rotazione) una carta in realta' fuori rotazione/solo Wild,
presentata come se fosse valida in quel mazzo - lo stesso tipo di errore "plausibile
ma falso" gia' documentato per altri campi nella guida di progetto, qui specifico
alla legalita' di formato.

CORREZIONE DEL 08/09/2026 (segnalata dall'utente, che ha notato che la cache locale
conteneva solo 3 set invece dei ~7 attesi): la prima versione di questo modulo usava
/api/v1/constructed/meta_period/latest/ e il suo campo "standard_legal_sets" come se
fosse l'elenco COMPLETO della rotazione Standard corrente. Verifica diretta della
risposta reale (fetch dal browser, non piu' un riassunto di WebFetch che aveva gia'
troncato/frainteso l'array durante l'esplorazione iniziale): quel campo contiene SOLO
i set legati al motivo specifico di quel meta-period (reason: "BALANCE_CHANGE"), cioe'
un changelog dell'ultimo cambiamento, non l'istantanea completa di cosa e' legale ora.
Nell'esempio verificato conteneva un solo set (ESCAPEFROM_VIOLET_HOLD) - la struttura
JSON sembrava quella giusta, il suo SIGNIFICATO no. Stesso principio gia' scritto nella
guida di progetto ("verificare sempre uno schema noto prima di costruirci sopra"), qui
applicato in modo tardivo.

Nessun endpoint pubblico di HSReplay restituisce l'elenco completo e aggiornato della
rotazione Standard (verificato: solo questo changelog parziale). La lista sotto e'
quindi MANTENUTA A MANO, verificata il 08/09/2026 incrociando due fonti:
1. hearthstone.wiki.gg/wiki/Standard_format (rotazione corrente, anno "Scarab" 2026 +
   anno precedente "Raptor" 2025, piu' Core - regola ufficiale: Standard include
   l'espansione dell'anno in corso, quella dell'anno precedente, ed Event/Core).
2. Cross-check contro data/raw/hearthstonejson/cards.json del progetto per risolvere i
   nomi in codici HearthstoneJSON esatti e disambiguare casi ambigui - es. "Across the
   Timeways" e' il set TIME_TRAVEL (183 carte, tra cui "Chronikar", "Twilight
   Timereaver"), NON TAVERNS_OF_TIME che ha 0 carte collezionabili ed e' quindi un set
   vuoto/inutilizzato con un nome simile.

DA AGGIORNARE A MANO quando ruota lo Standard (di norma annualmente, oltre alle uscite
di mini-espansioni durante l'anno): aggiungere il nuovo codice set, rimuovere quelli
usciti dalla rotazione (l'anno rotazionale piu' vecchio). Nessun meccanismo automatico
puo' farlo in modo affidabile con le fonti pubbliche trovate finora.
"""
from __future__ import annotations

import os
import re

import requests

META_PERIOD_URL = "https://hsreplay.net/api/v1/constructed/meta_period/latest/"
CACHE_PATH = os.path.join("data", "raw", "hsreplay", "standard_legal_sets.json")

# Elenco mantenuto a mano - vedi spiegazione e metodo di verifica nel docstring del
# modulo sopra. Verificato 08/09/2026.
STANDARD_LEGAL_SETS_MANUAL = {
    "CORE", "CORE_HIDDEN",          # Basic/Core Set, legale per sempre
    "EVENT",                        # carte da eventi stagionali (18 carte, ruotano come un mini-set)
    "CATACLYSM",                    # Marzo 2026 - anno rotazionale "Scarab"
    "ESCAPEFROM_VIOLET_HOLD",       # Luglio 2026 - anno rotazionale "Scarab"
    "EMERALD_DREAM",                # Marzo 2025, "Into the Emerald Dream" - anno rotazionale "Raptor"
    "THE_LOST_CITY",                # Luglio 2025, "The Lost City of Un'Goro" - anno rotazionale "Raptor"
    "TIME_TRAVEL",                  # Novembre 2025, "Across the Timeways" - anno rotazionale "Raptor"
}

# Il Basic Set (e la sua variante "nascosta" CORE_HIDDEN) e' legale in Standard per
# definizione permanente del gioco - lo forziamo comunque nell'insieme anche se
# l'endpoint non lo elencasse esplicitamente, per non rischiare falsi positivi
# assurdi (una carta base segnalata come "non Standard").
ALWAYS_STANDARD_LEGAL = {"CORE", "CORE_HIDDEN"}


def fetch_standard_legal_sets() -> set[str]:
    """Ritorna l'insieme dei codici set (stesso formato/valori del campo 'set' di
    HearthstoneJSON) attualmente legali in Standard. Base: la lista mantenuta a mano
    sopra (STANDARD_LEGAL_SETS_MANUAL), sempre presente - niente piu' un None da
    gestire a valle, visto che ora c'e' sempre un valore affidabile anche a rete
    spenta. In piu', prova (best-effort, MAI l'unica fonte: vedi correzione del
    08/09/2026 nel docstring del modulo) a recuperare l'endpoint HSReplay e unisce
    eventuali codici nuovi non ancora presenti nella lista a mano - non fa mai danno
    (un set in piu' unito non causa falsi negativi) e puo' fare da rete di sicurezza
    se la lista a mano non viene aggiornata in tempo dopo una rotazione."""
    sets = set(STANDARD_LEGAL_SETS_MANUAL)
    try:
        resp = requests.get(META_PERIOD_URL, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        extra = {s["code"] for s in payload.get("standard_legal_sets", []) if s.get("code")}
        sets |= extra
    except Exception:
        pass  # rete irraggiungibile o risposta inattesa - va bene, la lista a mano basta da sola
    return sets | ALWAYS_STANDARD_LEGAL


# Classi giocatore di Hearthstone (valore del campo 'cardClass' in HearthstoneJSON,
# stesso identificatore usato da deck_class in domain_data.py) mappate al nome che il
# Planner usa nei topic (in inglese, perche' cosi' sono scritti i deck_class nel
# dataset - vedi gli esempi nei topic generati, es. "Priest Standard", "Warlock Wild").
# "NEUTRAL" non e' una classe giocatore: le carte neutrali sono giocabili in QUALSIASI
# mazzo, quindi vanno sempre considerate valide a prescindere dalla classe del mazzo.
CLASS_NAME_TO_CODE = {
    "death knight": "DEATHKNIGHT",
    "demon hunter": "DEMONHUNTER",
    "druid": "DRUID",
    "hunter": "HUNTER",
    "mage": "MAGE",
    "paladin": "PALADIN",
    "priest": "PRIEST",
    "rogue": "ROGUE",
    "shaman": "SHAMAN",
    "warlock": "WARLOCK",
    "warrior": "WARRIOR",
}


def detect_deck_class(text: str) -> str | None:
    """Euristica minima sul topic/justification del post per capire di quale classe
    e' il mazzo discusso (stesso limite di detect_format() sotto: non c'e' un campo
    strutturato che colleghi il post al mazzo specifico - solo testo libero). Cerca il
    nome inglese della classe come sottostringa (i topic del Planner usano sempre il
    nome inglese, es. "Guida al Priest Standard..." - vedi deck_class nel dataset).
    Controlla "demon hunter"/"death knight" PRIMA delle classi a una parola per non
    confondere "hunter" dentro "demon hunter" con la classe Hunter (e la rimuove dal
    testo controllato dopo averla trovata, per lo stesso motivo).

    Ritorna None sia se NESSUNA classe e' riconoscibile, sia se PIU' di una classe
    distinta compare nel testo (es. un post che confronta due mazzi di classi diverse
    - "Rafaam Warlock e il Warrior Wild", caso reale osservato l'11/09/2026): scegliere
    arbitrariamente la prima trovata sarebbe attivamente sbagliato, non solo impreciso
    - un claim RAG legittimo sulla SECONDA classe menzionata verrebbe scartato da
    class_valid in research.py come "carta fuori classe", identico al problema gia'
    risolto per detect_format() con l'esito "misto" (vedi sotto). Il chiamante deve
    trattare None come "non applicare il filtro di classe", non come "nessuna classe
    nota" nel senso di "post senza classe" - stesso principio "meglio non verificare
    che verificare in modo sbagliato" gia' consolidato in questo modulo."""
    t = (text or "").lower()
    found: set[str] = set()
    for name in ("demon hunter", "death knight"):
        if name in t:
            found.add(CLASS_NAME_TO_CODE[name])
            t = t.replace(name, " ")
    for name, code in CLASS_NAME_TO_CODE.items():
        if name in t:
            found.add(code)
    if len(found) == 1:
        return found.pop()
    return None


def detect_format(text: str) -> str:
    """Euristica minima sul topic/justification del post per capire se si parla di un
    mazzo Standard o Wild (non c'e' un campo strutturato che colleghi il post del
    Planner al mazzo specifico usato per generarlo - vedi nota in research.py).

    Tre esiti possibili:
    - "wild" se il testo nomina solo "wild"
    - "standard" se nomina solo "standard", o nessuno dei due (assunzione piu' prudente,
      e' il formato della maggioranza dei mazzi nel dataset)
    - "misto" se nomina ENTRAMBI (es. un post "news" sui cambiamenti nel meta di
      Standard E Wild insieme, caso reale osservato il 10/09/2026) - qui forzare
      "standard" sarebbe attivamente sbagliato, non solo impreciso: scarterebbe come
      'fuori formato' claim su carte Wild-only che sono contenuto legittimo per questo
      tipo di post. Il chiamante deve trattare "misto" come "non applicare il filtro
      di legalita' Standard" (stesso principio gia' usato per detect_deck_class: meglio
      non verificare che verificare in modo sbagliato)."""
    t = (text or "").lower()
    has_wild = "wild" in t
    has_standard = "standard" in t
    if has_wild and has_standard:
        return "misto"
    if has_wild:
        return "wild"
    return "standard"


# Aggiunto il 16/09/2026 dopo un caso reale segnalato dall'utente: un post pianificato
# come "Analisi del meta Wild dopo l'ultimo aggiornamento di bilanciamento" (quindi un
# mazzo costruito, Standard/Wild - detect_format() sopra non conosce altro) ha finito
# per citare "Trinket" e "Dark Gift" come se fossero cambiamenti al meta Wild, quando
# in realta' sono meccaniche ESCLUSIVE di Battlegrounds (modalita' completamente
# diversa dal gioco costruito, verificato sulle note della patch 36.2.2 citata dal
# post stesso: la sezione Trinket/Dark Gift e' sotto "Battlegrounds Updates", non sotto
# le modifiche Standard/Wild). La causa non e' un formato/espansione sbagliati (quello
# lo controlla gia' format_valid per le carte RAG) ma una modalita' di gioco diversa
# nascosta dentro la STESSA pagina di patch notes (Blizzard pubblica sempre Standard/
# Wild e Battlegrounds nello stesso articolo) - search_web non ha modo di saperlo, e
# nessun controllo esistente lo intercetta perche' questo pipeline pianifica SOLO post
# su mazzi costruiti (il dataset del Planner viene da metastats/HSReplay, dati
# esclusivamente di gioco costruito - vedi guida di progetto), quindi qualunque
# menzione di questi termini in un claim e' gia' di per se' un segnale di modalita'
# sbagliata, non serve nemmeno sapere il formato rilevato del post specifico.
#
# Elenco volutamente MINIMO e MECCANICO (stesso principio di SOURCE_TIER_DOMAINS in
# search_tool.py: nessun giudizio semantico, solo termini che in Hearthstone non
# esistono FUORI da Battlegrounds) - "battlegrounds" stesso, piu' le due meccaniche
# del caso reale. Non include termini piu' ambigui (es. "hero power", "tavern") che
# hanno anche un significato nel gioco costruito o sarebbero troppo generici.
BATTLEGROUNDS_ONLY_KEYWORDS = ("battlegrounds", "trinket", "dark gift")


def mentions_battlegrounds_only(text: str) -> bool:
    """True se il testo (tipicamente un claim gia' estratto da Research) nomina un
    termine esclusivo di Battlegrounds (vedi BATTLEGROUNDS_ONLY_KEYWORDS sopra) - un
    segnale meccanico che quel claim riguarda la modalita' Battlegrounds e non il mazzo
    costruito di cui parla il post, da trattare come lo stesso tipo di problema di
    format_valid/class_valid (fuori dominio del post), non un errore di fatto sulla
    fonte in se'."""
    t = (text or "").lower()
    return any(kw in t for kw in BATTLEGROUNDS_ONLY_KEYWORDS)



# Correzione stilistica segnalata dall'utente l'11/09/2026: nella community italiana
# di Hearthstone il "metagame" competitivo si dice al MASCHILE ("il meta"), non al
# femminile ("la meta", che in italiano standard significherebbe "traguardo/obiettivo"
# - genere diverso, parola diversa nel significato inteso qui). Sia PLANNER_SYSTEM_PROMPT
# che FORMAT_SYSTEM_PROMPT usano gia' "il meta"/"nel meta" nel proprio testo (quindi il
# modello ha gia' un esempio corretto sotto gli occhi), ma qwen3:8b lo ha comunque
# scritto al femminile in piu' run reali - stesso limite di compliance testuale gia'
# documentato altrove in questo progetto. Regola di codice come rete di sicurezza,
# oltre a una regola esplicita aggiunta a entrambi i prompt.
_META_GENDER_FIXES = [
    (r"\bdella\b(?=\s+meta\b)", "del"),
    (r"\bnella\b(?=\s+meta\b)", "nel"),
    (r"\balla\b(?=\s+meta\b)", "al"),
    (r"\bsulla\b(?=\s+meta\b)", "sul"),
    (r"\bdalla\b(?=\s+meta\b)", "dal"),
    (r"\bquesta\b(?=\s+meta\b)", "questo"),
    (r"\bquella\b(?=\s+meta\b)", "quel"),
    (r"\buna\b(?=\s+meta\b)", "un"),
    (r"\ble\b(?=\s+meta\b)", "i"),
    (r"\bla\b(?=\s+meta\b)", "il"),
]


def fix_meta_gender(text: str) -> tuple[str, int]:
    """Corregge l'articolo/preposizione articolata quando precede DIRETTAMENTE la
    parola "meta" (es. "la meta" -> "il meta", "della meta" -> "del meta"), per la
    convenzione della community italiana di Hearthstone (maschile). Copre solo
    l'adiacenza diretta articolo+"meta": un caso con un aggettivo in mezzo (es. "la
    nuova meta") NON viene corretto di proposito - un regex piu' aggressivo rischia
    falsi positivi/frasi grammaticalmente peggiori, e questo progetto preferisce non
    correggere piuttosto che correggere in modo sbagliato o incompleto (stesso
    principio gia' applicato a detect_deck_class/detect_format sopra). Non e' un vero
    parser grammaticale: non sistema l'accordo di eventuali aggettivi/pronomi che nella
    stessa frase si riferiscono a "meta" ma non le sono adiacenti.

    Ritorna (testo_corretto, numero_di_sostituzioni) - il chiamante logga nel
    reasoning_trace solo se e' stato corretto davvero qualcosa."""
    if not text:
        return text, 0
    n_fixes = 0

    def _make_replacer(repl: str):
        def _replace(m: re.Match) -> str:
            nonlocal n_fixes
            n_fixes += 1
            matched = m.group(0)
            if matched[:1].isupper():
                return repl[:1].upper() + repl[1:]
            return repl
        return _replace

    for pattern, replacement in _META_GENDER_FIXES:
        text = re.sub(pattern, _make_replacer(replacement), text, flags=re.IGNORECASE)
    return text, n_fixes
