"""Regole sul formato di gioco (Standard vs Wild): in Wild sono legali tutte le
carte mai pubblicate, in Standard solo le espansioni in rotazione. L'elenco delle
espansioni Standard e' mantenuto a mano (nessun endpoint pubblico lo espone per
intero) e va aggiornato quando ruota lo Standard."""
from __future__ import annotations

import os
import re

import requests

META_PERIOD_URL = "https://hsreplay.net/api/v1/constructed/meta_period/latest/"
CACHE_PATH = os.path.join("data", "raw", "hsreplay", "standard_legal_sets.json")


STANDARD_LEGAL_SETS_MANUAL = {
    "CORE", "CORE_HIDDEN",          # Basic/Core Set, legale per sempre
    "EVENT",                        # carte da eventi stagionali (18 carte, ruotano come un mini-set)
    "CATACLYSM",                    # Marzo 2026 - anno rotazionale "Scarab"
    "ESCAPEFROM_VIOLET_HOLD",       # Luglio 2026 - anno rotazionale "Scarab"
    "EMERALD_DREAM",                # Marzo 2025, "Into the Emerald Dream" - anno rotazionale "Raptor"
    "THE_LOST_CITY",                # Luglio 2025, "The Lost City of Un'Goro" - anno rotazionale "Raptor"
    "TIME_TRAVEL",                  # Novembre 2025, "Across the Timeways" - anno rotazionale "Raptor"
}

# Basic Set (e CORE_HIDDEN) legale in Standard per definizione permanente - forzato
# comunque nell'insieme anche se l'endpoint non lo elencasse esplicitamente.
ALWAYS_STANDARD_LEGAL = {"CORE", "CORE_HIDDEN"}


def fetch_standard_legal_sets() -> set[str]:
    """Ritorna l'insieme dei codici set (formato HearthstoneJSON) attualmente legali
    in Standard: la lista unita a eventuali codici nuovi dall'endpoint HSReplay."""
    sets = set(STANDARD_LEGAL_SETS_MANUAL)
    try:
        resp = requests.get(META_PERIOD_URL, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        extra = {s["code"] for s in payload.get("standard_legal_sets", []) if s.get("code")}
        sets |= extra
    except Exception:
        pass  # rete irraggiungibile o risposta inattesa
    return sets | ALWAYS_STANDARD_LEGAL


# Nome inglese della classe (come compare nei topic del Planner) -> codice
# HearthstoneJSON. "NEUTRAL" non e' una classe giocatore: le carte neutrali sono
# giocabili in qualsiasi mazzo.
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
    """Euristica sul topic/justification del post per capire la classe del mazzo
    discusso (nessun campo strutturato collega il post al mazzo, solo testo libero).
    Ritorna None sia se nessuna classe e' riconoscibile sia se ne compare piu' di una
    (es. un post che confronta due mazzi) - il chiamante deve trattare None come "non
    applicare il filtro di classe", non come "post senza classe"."""
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
    """Euristica sul topic/justification del post per capire se si parla di un mazzo
    Standard o Wild. Tre esiti: "wild" (nomina solo wild), "standard" (nomina solo
    standard o nessuno dei due, assunzione piu' prudente), "misto" (nomina entrambi,
    es. un post "news" su Standard E Wild insieme) - il chiamante deve trattare
    "misto" come "non applicare il filtro di legalita' Standard"."""
    t = (text or "").lower()
    has_wild = "wild" in t
    has_standard = "standard" in t
    if has_wild and has_standard:
        return "misto"
    if has_wild:
        return "wild"
    return "standard"


# Termini che esistono SOLO in Battlegrounds (esclusi termini ambigui come "hero
# power"/"tavern", che hanno senso anche nel gioco costruito).
BATTLEGROUNDS_ONLY_KEYWORDS = ("battlegrounds", "trinket", "dark gift")


def mentions_battlegrounds_only(text: str) -> bool:
    """True se il testo nomina un termine esclusivo di Battlegrounds
    (BATTLEGROUNDS_ONLY_KEYWORDS sopra) - segnale che il claim riguarda quella
    modalita' e non il mazzo costruito di cui parla il post."""
    t = (text or "").lower()
    return any(kw in t for kw in BATTLEGROUNDS_ONLY_KEYWORDS)


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
    parola "meta" (es. "la meta" -> "il meta"), convenzione maschile della community
    italiana. Copre solo l'adiacenza diretta. Ritorna (testo_corretto, numero_di_sostituzioni)."""
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
