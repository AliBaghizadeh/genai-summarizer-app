# Configuration settings for the GenAI Summarizer App

# --- Model Names ---
EMBEDDING_MODEL = "models/text-embedding-004"
SUMMARY_MODEL = "models/gemini-1.5-pro"
EVALUATION_MODEL = "models/gemini-1.5-flash"

# --- Text Splitter Settings ---
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# --- ChromaDB Settings ---
DB_DIRECTORY = "./chroma_capstone_db"
DB_NAME = "capstone_rag_db"
# Setting for clearing DB on each run (True = clear, False = reuse)
# Set to False if you want persistence between app runs for the same data
CLEAR_DB_ON_RUN = True 

# --- Embedding Settings ---
# Corresponds to text-embedding-004 output dimension
EMBEDDING_DIM = 768
# Max batch size for embedding API calls
EMBEDDING_API_BATCH_SIZE = 100

# --- Evaluation Settings ---
# Max context length to pass to the evaluation model
EVALUATION_MAX_CONTEXT_LEN = 8000
# Evaluation model temperature (lower for more deterministic results)
EVALUATION_TEMPERATURE = 0.1

# --- RAG Settings ---
# Default number of chunks to retrieve (can be overridden by UI slider)
DEFAULT_K_RESULTS = 10 

# --- LangGraph Settings ---
# Recursion limit for the graph execution
AGENT_RECURSION_LIMIT = 15

# --- Web Scraping Settings ---
REQUESTS_TIMEOUT = 20 # seconds
REQUESTS_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

print("Config loaded.") # Optional: indicate config is parsed 