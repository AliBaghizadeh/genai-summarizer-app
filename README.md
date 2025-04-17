# GenAI Document Summarizer & Evaluator

## Overview

This project, developed as a capstone for the Google 5-Day GenAI Intensive Course, addresses the challenge of understanding and synthesizing information from numerous technical documents like research papers (PDFs) and web articles. Manually processing these diverse sources is time-consuming.

This Streamlit application leverages Google's Gemini models and vector databases (ChromaDB) to:
1.  Ingest text from uploaded PDFs and provided web URLs.
2.  Allow users to ask a specific query about the documents.
3.  Retrieve the most relevant information using semantic search (RAG).
4.  Generate a structured summary answering the query using Gemini 1.5 Pro.
5.  **Provide an automated quality assessment** of the generated summary using Gemini 1.5 Flash, adding a layer of trust and reliability.

The goal is to provide a tool that saves time, improves learning, and helps professionals quickly grasp key information from dense technical material.

## Features

*   **Multi-Source Ingestion:** Handles both PDF uploads and web URLs.
*   **Retrieval-Augmented Generation (RAG):** Uses ChromaDB vector storage and Gemini embeddings (`text-embedding-004`) for efficient semantic retrieval of relevant text chunks.
*   **Structured Summarization:** Leverages Gemini 1.5 Pro to generate summaries in a structured JSON format (containing the answer and source list).
*   **Automated Evaluation:** Employs Gemini 1.5 Flash to automatically evaluate the generated summary's faithfulness and completeness based on the retrieved context.
*   **Interactive UI:** Built with Streamlit for easy document upload, query input, and results display.

## Workflow

The application follows this pipeline:

1.  **User Query:** The user provides a question about the documents.
2.  **Document Loading:** PDFs are uploaded, and web URLs are provided. Text is extracted from both sources.
3.  **Chunking:** The extracted text is split into smaller, overlapping chunks.
4.  **Embedding:** Each chunk is converted into a numerical vector (embedding) using the Gemini embedding model, capturing its semantic meaning.
5.  **Vector Storage:** Embeddings and associated metadata are stored in a persistent ChromaDB collection.
6.  **Retrieval:** The user's query is embedded, and ChromaDB is searched to find the most semantically similar text chunks (`k` chunks, configurable via UI).
7.  **Summarization (LLM 1):** The retrieved chunks are passed as context to Gemini 1.5 Pro, which generates a structured JSON summary answering the query based *only* on this context.
8.  **Evaluation (LLM 2):** The generated summary and the retrieved context are passed to Gemini 1.5 Flash, which evaluates the summary's quality (faithfulness, completeness, etc.).
9.  **Output:** The structured summary and the evaluation result are displayed to the user via the Streamlit interface.

![Workflow Chart](path/to/your/Flowchart.png)

## GenAI Capabilities Demonstrated

This project utilizes several core GenAI techniques:

*   **Document Understanding:** Parsing text from PDFs (`PyMuPDF`) and web pages (`BeautifulSoup`).
*   **Embeddings:** Generating semantic vectors using Google's `text-embedding-004`.
*   **Vector Database:** Storing and querying embeddings with ChromaDB.
*   **Retrieval-Augmented Generation (RAG):** The core retrieve-then-generate pattern.
*   **Large Language Models (LLMs):** Using Gemini 1.5 Pro for generation and Gemini 1.5 Flash for evaluation.
*   **Controlled Generation (JSON Mode):** Instructing Gemini Pro to output structured JSON.
*   **Gen AI Evaluation:** Programmatically assessing LLM output quality using another LLM.
*   **(Potentially Few-Shot Prompting):** Depending on the final prompt structure used for summarization.
*   **(Potentially Function Calling/Agents):** Depending on how the evaluation step is invoked within the LangGraph structure (the description mentioned both direct calls and tools).

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\Activate.ps1

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure your `requirements.txt` file is accurate and includes `streamlit`, `google-generativeai`, `langchain`, `langgraph`, `chromadb`, `PyMuPDF`, `requests`, `beautifulsoup4`, `numpy`, etc., preferably with version numbers, e.g., `numpy==1.26.4`)*

## Configuration

*   **Google API Key:** You need a valid Google API Key with the "Generative Language API" enabled and billing set up for the associated Google Cloud project.
*   **Provide Key:** When you run the app, paste your Google API Key into the designated field in the Streamlit sidebar. For deployment, consider using Streamlit Secrets (`.streamlit/secrets.toml`).

## How to Run

1.  Ensure your virtual environment is activated.
2.  Navigate to the project directory in your terminal.
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4.  The application should open automatically in your web browser.

## File Structure
```
project-folder/
├── app.py # Streamlit UI logic
├── pipeline.py # Core RAG+Eval logic
├── config.py # Constants and settings
├── requirements.txt # Dependencies
├── chroma_capstone_db/ # Created by ChromaDB 
└── README.md # This file
```

## Limitations

*   **Retrieval Quality:** The quality of the summary depends heavily on whether the vector search retrieves truly relevant *and explanatory* chunks. Sometimes, related but non-informative chunks might be retrieved.
*   **Context Window vs. `k`:** Retrieving a very large number of chunks (`k`) to ensure completeness increases processing time and API costs.
*   **Evaluation Accuracy:** The automated evaluation by Gemini Flash provides a useful quality signal but is still an LLM output and may not be perfectly accurate or nuanced.
*   **LLM Faithfulness:** While prompted to stick to the context, the summarization LLM (Gemini Pro) could still potentially hallucinate or misinterpret complex information.
*   **Format Handling:** Complex PDF layouts or dynamic web content might not be parsed perfectly.

## Future Work

*   **Multi-Document Comparison:** Extend functionality to compare/contrast information across sources.
*   **Chatbot Interface:** Allow for conversational follow-up questions instead of single-shot queries.
*   **Integration:** Connect to research tools like reference managers or note-taking apps.
*   **Enhanced Evaluation/Refinement:** Incorporate more sophisticated metrics or allow the primary LLM agent to revise its summary based on the evaluation feedback.
*   **Hybrid Retrieval:** Combine vector search with traditional keyword search.
