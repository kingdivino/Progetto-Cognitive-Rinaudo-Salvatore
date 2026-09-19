# Blogger Support Agent — Hearthstone

Progetto per CCAI 2025-26. Agente LangGraph che supporta la scrittura di un blog su Hearthstone: suggerisce topic, verifica fonti, scrive bozze di post con revisione umana, mantiene un Knowledge Graph editoriale.

## Setup (prima volta)

### 1. Python e virtual environment
Serve Python 3.11+. Apri un terminale nella cartella del progetto:

```bash
python -m venv venv
```

Attiva l'ambiente:
- Windows (PowerShell): `venv\Scripts\Activate.ps1`
- Windows (cmd): `venv\Scripts\activate.bat`
- macOS/Linux: `source venv/bin/activate`

Poi installa le dipendenze:

```bash
pip install -r requirements.txt
```

### 2. Ollama (modello locale)
Scarica Ollama da https://ollama.com, installalo, poi da terminale:

```bash
ollama pull llama3.1:8b
```

Verifica che sia in esecuzione: apri http://localhost:11434 nel browser, dovresti vedere "Ollama is running".

### 3. Neo4j Desktop (Knowledge Graph)
Scarica Neo4j Desktop da https://neo4j.com/download/, installalo, crea un nuovo progetto e un nuovo database locale (imposta una password, la userai nel file `.env`), avvialo.

### 4. Chiavi API
Copia `.env.example` in un nuovo file chiamato `.env` e compila:
- `TAVILY_API_KEY` — registrati gratis su https://tavily.com
- `LANGSMITH_API_KEY` — registrati gratis su https://smith.langchain.com
- `NEO4J_PASSWORD` — la password che hai scelto al punto 3

Il file `.env` non va mai condiviso o caricato su git (è già escluso).

### 5. Verifica che tutto funzioni

```bash
python notebooks/10_test_kg_update.py
```

Esegue il grafo completo end-to-end (richiede Ollama e Neo4j avviati, indice RAG gia' costruito).

## Struttura del progetto

- `src/agent/` — il grafo LangGraph (nodi, stato, edge)
- `src/tools/` — i tool dell'agente (search, RAG, knowledge graph, classificatore fine-tuned)
- `src/kg/` — codice di interazione con Neo4j
- `data/raw/` — dati grezzi scaricati (card data, dataset per fine-tuning)
- `data/processed/` — dati puliti/trasformati
- `notebooks/` — script di scraping/feature engineering/fine-tuning e test end-to-end del grafo
- `tests/` — test dei singoli componenti

## Stato del progetto

Vedi i documenti nel progetto Claude collegato ("Progetto cognitive") per specifiche complete e roadmap.
