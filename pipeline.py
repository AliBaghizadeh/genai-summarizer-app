# pipeline.py: Core backend logic for the RAG + Evaluation pipeline

import os
import json
import time
import numpy as np

# LangChain & Google GenAI specific
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types

# ChromaDB specific imports
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

# LangGraph specific
from typing import TypedDict, Annotated, Sequence, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# --- ADD Imports for loading functions (ensure they are here) ---
import fitz # PyMuPDF
import requests
from bs4 import BeautifulSoup
# --- END ADD Imports ---

# Import constants from config file
import config

# --- ADD Document Loading Functions (ensure they are here) --- 
def load_pdf_text(uploaded_file_bytes, filename="unknown.pdf") -> str:
    """
    Extracts text content from PDF bytes.
    Args:
        uploaded_file_bytes: Bytes content of the PDF file.
        filename: Original filename for error messages.
    Returns:
        A single string containing all extracted text, or empty string on error.
    """
    try:
        doc = fitz.open(stream=uploaded_file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        print(f"Successfully loaded text from PDF: {filename}")
        return text
    except Exception as e:
        print(f"ERROR processing PDF {filename}: {e}")
        return "" # Return empty string on error

def load_web_text(url: str) -> str:
    """
    Extracts the main textual content from a given URL using requests and BeautifulSoup.
    """
    print(f"Attempting to load web text from: {url}")
    try:
        response = requests.get(url, timeout=config.REQUESTS_TIMEOUT, headers=config.REQUESTS_HEADERS)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"Warning: Skipping URL {url} - Content type is not HTML ({content_type})")
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        print(f"Successfully loaded text from URL: {url} (Length: {len(text)})")
        return text
    except requests.exceptions.RequestException as e:
        print(f"ERROR fetching URL {url}: {e}")
        return ""
    except Exception as e:
        print(f"ERROR processing URL {url}: {e}")
        return ""
# --- END ADD Document Loading Functions ---

# --- Core Logic Implementation ---

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for ChromaDB using Google Gemini API.
    Relies on genai.configure(api_key=...) being called beforehand.
    """
    def __init__(self):
        super().__init__()
        # Ensure client is configured globally via genai.configure()

    def __call__(self, input_texts: Documents) -> Embeddings:
        print(f"Embedding {len(input_texts)} texts...") # Use print for backend logs
        all_embeddings = []
        total_batches = (len(input_texts) + config.EMBEDDING_API_BATCH_SIZE - 1) // config.EMBEDDING_API_BATCH_SIZE

        for i in range(0, len(input_texts), config.EMBEDDING_API_BATCH_SIZE):
            batch_num = i // config.EMBEDDING_API_BATCH_SIZE + 1
            print(f"Embedding batch {batch_num}/{total_batches}...")
            batch = input_texts[i : i + config.EMBEDDING_API_BATCH_SIZE]
            try:
                response = genai.embed_content(
                    model=config.EMBEDDING_MODEL,
                    content=batch,
                    task_type="retrieval_document"
                )
                batch_embeddings = response['embedding']
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"ERROR embedding batch {batch_num}: {e}")
                print(f"Adding dummy embeddings for failed batch {batch_num}.")
                all_embeddings.extend([[0.0] * config.EMBEDDING_DIM] * len(batch))

        print(f"Generated {len(all_embeddings)} embeddings.")
        if not all_embeddings:
             return []
        if len(all_embeddings[0]) != config.EMBEDDING_DIM:
            print(f"FATAL Error: Embeddings have incorrect dimension. Expected {config.EMBEDDING_DIM}, got {len(all_embeddings[0])}.")
            return []
        return all_embeddings

# --- Evaluation Function ---
def run_evaluation(summary_json_str: str, context: str, api_key: str) -> str:
    """
    Evaluates a summary JSON string against its original context using Gemini Flash.
    """
    if not api_key:
        return "Evaluation skipped: Google API Key not provided."

    print("Running summary evaluation with Gemini Flash...")
    try:
        eval_prompt = f"""
        Please evaluate the following summary based *only* on the provided context. Assess its quality based on these criteria:
        1.  **Faithfulness:** Does the summary accurately represent information present ONLY in the context? Note any contradictions or unsupported claims.
        2.  **Completeness:** Does the summary cover the main points relevant to the user's likely query (as implied by the context) found within the provided context?
        3.  **Conciseness:** Is the summary presented clearly and without unnecessary repetition?
        4.  **Reference Check:** Does the 'sources' list in the summary accurately reflect the sources mentioned in the context? (A list of sources is usually provided alongside the context).

        Provide a brief overall assessment (1-2 sentences) addressing these criteria.

        CONTEXT:
        ---
        {context[:config.EVALUATION_MAX_CONTEXT_LEN]}
        ---
        GENERATED SUMMARY (JSON String):
        ---
        {summary_json_str}
        ---
        EVALUATION:
        """
        eval_llm = ChatGoogleGenerativeAI(
            model=config.EVALUATION_MODEL,
            google_api_key=api_key,
            temperature=config.EVALUATION_TEMPERATURE
        )
        response = eval_llm.invoke(eval_prompt)
        print("Evaluation complete.")
        return response.content
    except Exception as e:
        print(f"ERROR during summary evaluation: {e}")
        return f"Evaluation failed: {e}"

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage | ToolMessage], add_messages]
    context_string: str | None = None
    initial_summary_json: dict | None = None

# --- Agent Execution Pipeline Function ---
def run_agent_pipeline(google_api_key, raw_docs_list, query, k_results):
    """Sets up and runs the full RAG + Evaluation pipeline."""

    # --- 1. Configure API ---
    try:
        genai.configure(api_key=google_api_key)
        print("Google API Key configured.")
    except Exception as e:
        print(f"ERROR: Failed to configure Google API Key: {e}")
        return {"error": "API Key Configuration Failed"}

    # --- 2. Initialize Splitter ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

    # --- 3. Chunk Documents ---
    print("Chunking documents...")
    chunked_documents = []
    for doc in raw_docs_list:
        if 'text' in doc and isinstance(doc['text'], str) and doc['text'].strip():
            try:
                chunks = splitter.split_text(doc['text'])
                for i, chunk_text in enumerate(chunks):
                    source_name = doc['source'][:200]
                    chunk_id = f"{source_name}_chunk_{i}"
                    chunked_documents.append({
                        'text': chunk_text,
                        'metadata': {
                            'source': source_name,
                            'type': doc['type'],
                            'chunk_id': chunk_id
                        }
                    })
            except Exception as e:
                 print(f"Warning: Error splitting document {doc.get('source', 'Unknown')}: {e}")
        else:
            print(f"Warning: Skipping document with missing/empty text: {doc.get('source', 'Unknown')}")

    if not chunked_documents:
        print("ERROR: Failed to create any document chunks.")
        return {"error": "Chunking Failed"}
    print(f"Created {len(chunked_documents)} chunks.")

    # --- 4. Setup Embedding Function ---
    embedding_function = GeminiEmbeddingFunction()

    # --- 5. Setup ChromaDB ---
    print(f"Setting up vector database ({config.DB_NAME})...")
    try:
        chroma_client = chromadb.PersistentClient(path=config.DB_DIRECTORY)
        print(f"Using persistent DB at: {os.path.abspath(config.DB_DIRECTORY)}")
        if config.CLEAR_DB_ON_RUN:
            try:
                print(f"Attempting to delete existing collection: {config.DB_NAME}")
                chroma_client.delete_collection(name=config.DB_NAME)
                print(f"Cleared existing collection: {config.DB_NAME}")
                time.sleep(1)
            except Exception as e:
                print(f"Collection {config.DB_NAME} likely did not exist (delete error: {e}). Creating fresh.")
        else:
            print(f"Reusing existing DB collection: {config.DB_NAME}")

        db_collection = chroma_client.get_or_create_collection(
            name=config.DB_NAME,
            embedding_function=embedding_function
        )
        print(f"ChromaDB collection '{config.DB_NAME}' ready.")
    except Exception as e:
        print(f"ERROR: Failed to initialize/clear ChromaDB: {e}")
        return {"error": f"ChromaDB Initialization Failed: {e}"}

    # --- 6. Add documents to ChromaDB ---
    # Check if collection is empty before adding, if not clearing
    if config.CLEAR_DB_ON_RUN or db_collection.count() == 0:
        print("Adding documents to vector database...")
        doc_texts = [doc['text'] for doc in chunked_documents]
        doc_ids = [doc['metadata']['chunk_id'] for doc in chunked_documents]
        doc_metadatas = [
            {'source': str(doc['metadata']['source']), 'type': str(doc['metadata']['type'])}
            for doc in chunked_documents
        ]
        try:
            batch_size = config.EMBEDDING_API_BATCH_SIZE # Align with embedding batch size
            total_batches = (len(doc_texts) + batch_size - 1) // batch_size
            print(f"Adding {len(doc_texts)} docs in {total_batches} batches...")
            for i in range(0, len(doc_texts), batch_size):
                current_batch_num = i//batch_size + 1
                print(f"Adding batch {current_batch_num}/{total_batches}...")
                ids_batch = doc_ids[i:i+batch_size]
                docs_batch = doc_texts[i:i+batch_size]
                meta_batch = doc_metadatas[i:i+batch_size]
                if not ids_batch or not docs_batch or not meta_batch:
                     print(f"Warning: Skipping empty batch component at index {i}")
                     continue
                db_collection.add(ids=ids_batch, documents=docs_batch, metadatas=meta_batch)
                time.sleep(0.1)
            print(f"Finished adding documents! Final count: {db_collection.count()}")
        except Exception as e:
            print(f"ERROR: Failed to add documents to ChromaDB: {e}")
            return {"error": f"ChromaDB Add Failed: {e}"}
    else:
         print(f"Skipping document add, collection already contains {db_collection.count()} items.")

    # --- 7. Define LangGraph Nodes (Local Scope) ---
    def embed_and_search_node_local(state: AgentState) -> AgentState:
        print("--- Running: embed_and_search_node ---")
        messages = state["messages"]
        query = messages[-1].content
        print(f"Searching vector database for: '{query}' (k={k_results})...")
        try:
            results = db_collection.query(
                query_texts=[query],
                n_results=int(k_results),
                include=['documents', 'metadatas']
            )
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            print(f"Retrieved {len(documents)} relevant document chunks.")

            if not documents:
                message = AIMessage(content="Could not find relevant documents to answer the query.")
                state["context_string"] = ""
            else:
                context_items = []
                unique_sources_list = set()
                for doc, meta in zip(documents, metadatas):
                    source_name = meta.get('source', 'Unknown')
                    unique_sources_list.add(source_name)
                    context_items.append(doc)
                context_str = "\n\n---\n\n".join(context_items)
                sources_for_prompt = sorted(list(unique_sources_list))
                state["context_string"] = context_str

                prompt_instruction = f"""You are an AI assistant... Use ONLY this provided text content... Do not attempt to access external websites or files...
User Query: '{query}'
--- START OF PROVIDED TEXT CONTEXT ---
{context_str}
--- END OF PROVIDED TEXT CONTEXT ---
Based *only* on the text provided above, generate a JSON object... 'answer' and 'sources'.
- 'answer' field...
- 'sources' field MUST be a list of strings containing ALL unique source document names... The available sources are: {sources_for_prompt}. List only these names.
Respond ONLY with the valid JSON object.
"""
                message = AIMessage(content=prompt_instruction, name="EmbedAndSearch")
            state["messages"].append(message)
        except Exception as e:
             print(f"ERROR during search node: {e}")
             state["messages"].append(AIMessage(content=f"Error during search: {e}"))
             state["context_string"] = ""
        return state

    def generate_summary_node_local(state: AgentState) -> dict:
        print("--- Running: generate_summary_node ---")
        messages = state['messages']
        if "Error during search:" in messages[-1].content or not state.get("context_string"):
             print("Warning: Skipping summary generation due to search error or no context found.")
             return {"initial_summary_json": {"answer": "Summary generation skipped due to search error or no context.", "sources": []}}
        print("Generating summary with Gemini Pro...")
        try:
            llm = ChatGoogleGenerativeAI(
                model=config.SUMMARY_MODEL, google_api_key=google_api_key,
                generation_config={"response_mime_type": "application/json"}
            )
            response_message = llm.invoke(messages)
            raw_content = response_message.content
            if raw_content.strip().startswith("```json"):
                raw_content = raw_content.strip()[len("```json"):].strip()
            if raw_content.strip().endswith("```"):
                raw_content = raw_content.strip()[:-len("```")].strip()
            summary_json = json.loads(raw_content)
            print("Summary generated.")
            return {"initial_summary_json": summary_json}
        except json.JSONDecodeError as json_e:
             print(f"ERROR decoding LLM JSON response: {json_e}")
             print(f"Raw Content: {raw_content}")
             return {"initial_summary_json": {"answer": f"Error: Failed to parse LLM response.", "sources": [], "raw_error_output": raw_content}}
        except Exception as e:
            print(f"ERROR during summary generation: {e}")
            raw_err_content = getattr(response_message, 'content', 'No response content') if 'response_message' in locals() else 'No response object'
            return {"initial_summary_json": {"answer": f"Error generating summary: {e}", "sources": [], "raw_error_output": raw_err_content}}

    def evaluate_node_local(state: AgentState) -> dict:
        print("--- Running: evaluate_node ---")
        summary_data = state.get("initial_summary_json")
        full_context = state.get("context_string", "")
        if summary_data and 'error' not in summary_data and summary_data.get('answer'):
            print("Calling evaluation function...")
            try:
                summary_json_str = json.dumps(summary_data)
                eval_result = run_evaluation(
                    summary_json_str=summary_json_str,
                    context=full_context,
                    api_key=google_api_key
                )
                return {"messages": [AIMessage(content=f"EVALUATION_RESULT:\n{eval_result}")]}
            except Exception as e:
                 print(f"ERROR calling evaluation function: {e}")
                 return {"messages": [AIMessage(content=f"EVALUATION_RESULT:\nEvaluation function failed: {e}")]}
        else:
            print("Warning: Skipping evaluation because summary was not generated successfully.")
            return {"messages": [AIMessage(content="EVALUATION_RESULT:\nSkipped due to summary generation error or missing answer.")]}

    # --- 8. Define and Compile LangGraph Agent ---
    print("Compiling processing agent graph...")
    try:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("embed_and_search", embed_and_search_node_local)
        graph_builder.add_node("generate_summary", generate_summary_node_local)
        graph_builder.add_node("evaluate_summary", evaluate_node_local)
        graph_builder.set_entry_point("embed_and_search")
        graph_builder.add_edge("embed_and_search", "generate_summary")
        graph_builder.add_edge("generate_summary", "evaluate_summary")
        graph_builder.add_edge("evaluate_summary", END)
        agent_executor = graph_builder.compile()
        print("Agent compiled successfully.")
    except Exception as e:
        print(f"ERROR: Failed to compile LangGraph agent: {e}")
        return {"error": f"Agent Compilation Failed: {e}"}

    # --- 9. Invoke Agent ---
    print("Running agent inference...")
    initial_state = {"messages": [HumanMessage(content=query)]}
    try:
        start_time = time.time()
        final_state = agent_executor.invoke(initial_state, {"recursion_limit": config.AGENT_RECURSION_LIMIT})
        end_time = time.time()
        print(f"Agent execution took {end_time - start_time:.2f} seconds.")
        return final_state
    except Exception as e:
        print(f"ERROR during agent execution: {e}")
        error_message = f"Agent Execution Failed: {e}"
        return {"error": error_message, "messages": initial_state.get("messages", [])} 