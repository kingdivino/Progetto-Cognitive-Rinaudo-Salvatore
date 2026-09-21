Blogger Support Agent - Hearthstone

Progetto per CCAI 2025-26. Agente LangGraph che supporta la scrittura di un blog su Hearthstone: suggerisce topic, verifica fonti, scrive bozze di post con revisione umana, mantiene un Knowledge Graph editoriale.


INSTALLAZIONE

1. Python e virtual environment

Serve Python 3.11 o superiore. Apri un terminale nella cartella del progetto e crea l'ambiente virtuale:

python -m venv venv

Attiva l'ambiente:

Windows PowerShell: venv\Scripts\Activate.ps1
Windows cmd: venv\Scripts\activate.bat
macOS/Linux: source venv/bin/activate

Installa le dipendenze:

pip install -r requirements.txt


2. Ollama (modello locale)

Scarica e installa Ollama da https://ollama.com, poi da terminale scarica il modello:

ollama pull qwen3:8b

Verifica che sia in esecuzione aprendo http://localhost:11434 nel browser: deve comparire "Ollama is running".


3. Neo4j Desktop (Knowledge Graph)

Scarica Neo4j Desktop da https://neo4j.com/download/, installalo, crea un nuovo progetto e un nuovo database locale (imposta una password, la userai nel file .env), avvialo.


4. Chiavi API

Copia .env.example in un nuovo file chiamato .env e compila:

TAVILY_API_KEY: 
LANGSMITH_API_KEY: 
NEO4J_PASSWORD: la password scelta al punto 3

5. Verifica che tutto funzioni

python notebooks/10_test_kg_update.py

Esegue il grafo completo end-to-end (richiede Ollama e Neo4j avviati, indice RAG gia' costruito).


STRUTTURA DEL PROGETTO

src/agent/     il grafo LangGraph (nodi, stato, edge)
src/tools/     i tool dell'agente (search, RAG, knowledge graph, classificatore fine-tuned)
src/kg/        codice di interazione con Neo4j
data/raw/      dati grezzi scaricati (card data, dataset per fine-tuning)
data/processed/  dati puliti/trasformati
notebooks/     script di scraping/feature engineering/fine-tuning e test end-to-end del grafo
tests/         test dei singoli componenti

