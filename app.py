import streamlit as st
import os
import json
import time

# --- Set Page Config FIRST --- 
st.set_page_config(layout="wide")

# --- Import backend logic from pipeline.py --- 
# Use try-except for robustness in case files are missing initially
try:
    from pipeline import run_agent_pipeline, load_pdf_text, load_web_text
    pipeline_imported = True
except ImportError as e:
    # Store error message, display *after* set_page_config
    pipeline_import_error = f"Error importing from pipeline.py: {e}. Make sure pipeline.py exists and has the required functions."
    pipeline_imported = False
    # Define dummy functions so the rest of the UI code doesn't immediately break
    def load_pdf_text(f): return ""
    def load_web_text(u): return ""
    def run_agent_pipeline(*args, **kwargs): return {"error": "Pipeline module not found or failed to import"}
else: # No error occurred
    pipeline_import_error = None

# --- Import constants from config.py ---
try:
    import config
    config_imported = True
except ImportError as e:
    # Store error message
    config_import_error = f"Error importing from config.py: {e}. Make sure config.py exists."
    config_imported = False
    # Define dummy config values
    class DummyConfig:
        DEFAULT_K_RESULTS = 10
        CLEAR_DB_ON_RUN = True
    config = DummyConfig()
else: # No error occurred
    config_import_error = None

# --- Streamlit App UI ---
# Display title AFTER set_page_config
st.title("📄🤖 GenAI Document Summarizer & Evaluator")

# Display import errors now if they occurred
if pipeline_import_error:
    st.error(pipeline_import_error, icon="🚨")
if config_import_error:
    st.error(config_import_error, icon="🚨")

# --- Sidebar for API Keys and Config ---
with st.sidebar:
    st.header("🔑 API Keys")
    # Remove the attempt to read from st.secrets for local running
    # default_google_key = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, "secrets") else ""
    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        help="Required for Gemini models.",
        # value=default_google_key # Remove default value from secrets
        )

    st.header("⚙️ Configuration")
    k_retrieval = st.slider(
        "Chunks to Retrieve (k)",
        min_value=5,
        max_value=100,
        value=config.DEFAULT_K_RESULTS, # Uses imported or dummy config
        step=5,
        help="Number of text chunks to retrieve based on your query."
    )
    clear_db = st.checkbox(
        "Clear Database on Run",
        value=config.CLEAR_DB_ON_RUN, # Uses imported or dummy config
        help="Check this to force reprocessing and re-embedding all documents each time. Uncheck to reuse existing database.",
        key="clear_db_checkbox"
        )

# --- Main Input Area --- (Remains the same)
st.header("1. Upload Documents")
uploaded_pdfs = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDF documents."
)
st.header("2. Add Web Links")
web_links_text = st.text_area(
    "Enter Web URLs (one per line)",
    height=150,
    help="Paste URLs of web pages you want to include."
)
st.header("3. Enter Your Query")
user_query = st.text_input(
    "What do you want to summarize or ask about these documents?",
    placeholder="e.g., Explain attention mechanisms in vision transformers"
)

# --- Processing Trigger --- (Remains the same)
st.header("4. Run Analysis")
run_button = st.button("Generate Summary & Evaluation")

# --- Output Area --- (Remains the same)
st.divider()
st.header("📈 Results")

output_placeholder = st.container()

if run_button:
    # Input validation
    # Add check for import success
    if not pipeline_imported or not config_imported:
        st.error("❌ Cannot run analysis because backend code (pipeline.py or config.py) failed to load.")
    elif not google_api_key:
        st.error("❌ Please enter your Google API Key in the sidebar or set it via Streamlit Secrets (GOOGLE_API_KEY).")
    # (Rest of input validation and run logic remains the same)
    elif not (uploaded_pdfs or web_links_text):
        st.error("❌ Please upload at least one PDF or enter at least one Web URL.")
    elif not user_query:
        st.error("❌ Please enter your query.")
    else:
        output_placeholder.empty()
        with output_placeholder:
            with st.spinner("Processing documents and running analysis... This may take several minutes."):
                st.info("Loading and processing input documents...")
                raw_docs_list = []
                if uploaded_pdfs:
                    st.write(f"Reading {len(uploaded_pdfs)} PDF file(s)...")
                    for pdf_file in uploaded_pdfs:
                        # Use imported function, pass bytes
                        pdf_text = load_pdf_text(pdf_file.getvalue(), filename=pdf_file.name)
                        if pdf_text:
                            raw_docs_list.append({"source": pdf_file.name, "text": pdf_text, "type": "pdf"})
                if web_links_text:
                    urls = [url.strip() for url in web_links_text.splitlines() if url.strip()]
                    st.write(f"Reading {len(urls)} web link(s)...")
                    for url in urls:
                        web_text = load_web_text(url) # Uses imported or dummy function
                        if web_text:
                            raw_docs_list.append({"source": url, "text": web_text, "type": "web"})

                if not raw_docs_list:
                     st.error("Could not load text from any provided sources, or all sources failed.", icon="🚨")
                     st.stop()

                st.success(f"Successfully loaded text from {len(raw_docs_list)} sources.")
                st.info(f"Starting agent pipeline... (Retrieving k={k_retrieval} chunks) This can take time.")

                if config_imported:
                     config.CLEAR_DB_ON_RUN = st.session_state.clear_db_checkbox

                final_state = run_agent_pipeline( # Uses imported or dummy function
                    google_api_key=google_api_key,
                    raw_docs_list=raw_docs_list,
                    query=user_query,
                    k_results=k_retrieval
                )

                st.divider()
                if final_state and isinstance(final_state, dict) and 'error' in final_state:
                    st.error(f"Pipeline Error: {final_state['error']}", icon="🚨")
                    if 'messages' in final_state:
                         with st.expander("Show Agent Messages on Error"):
                             display_messages = [msg.dict() if hasattr(msg, 'dict') else msg for msg in final_state['messages']]
                             st.json(display_messages)
                    st.stop()
                elif not final_state or not isinstance(final_state, dict):
                     st.error("Pipeline execution failed to return a valid state.", icon="🚨")
                     st.stop()
                else:
                    st.success("Agent pipeline finished successfully!")
                    # (Rest of result display logic remains the same)
                    summary_data = final_state.get("initial_summary_json", {})
                    messages = final_state.get("messages", [])
                    st.subheader("Summary")
                    answer_text = summary_data.get('answer', "*(Summary content not found)*")
                    if "Error generating summary:" in answer_text or "Error: Failed to parse LLM response." in answer_text or "Summary generation skipped" in answer_text:
                         st.warning(answer_text, icon="⚠️")
                         if summary_data.get("raw_error_output"):
                              with st.expander("Show Raw LLM Output (Error)"):
                                   st.code(summary_data["raw_error_output"], language=None)
                    else:
                         st.markdown(answer_text)
                    if 'sources' in summary_data and summary_data['sources']:
                        st.subheader("References")
                        if isinstance(summary_data['sources'], list) and summary_data['sources']:
                            ref_markdown = ""
                            unique_refs = sorted(list(set(summary_data['sources'])))
                            for ref in unique_refs:
                                escaped_ref = str(ref).replace("<", "&lt;").replace(">", "&gt;")
                                ref_markdown += f"- <font color='blue'>{escaped_ref}</font>\n"
                            st.markdown(ref_markdown, unsafe_allow_html=True)
                        elif summary_data['sources']:
                             st.warning(f"*(Sources format incorrect: {summary_data['sources']})*")
                    st.markdown("## <font color='red'><b>--- Evaluation Result ---</b></font>", unsafe_allow_html=True)
                    evaluation_result_text = "*(Evaluation not found in final messages)*"
                    eval_msg_found = False
                    if messages:
                        for msg in reversed(messages):
                            if isinstance(msg, dict): msg_content = msg.get('content', '')
                            elif hasattr(msg, 'content'): msg_content = msg.content
                            else: msg_content = str(msg)
                            if "EVALUATION_RESULT:\n" in msg_content:
                                evaluation_result_text = msg_content.split("EVALUATION_RESULT:\n", 1)[1]
                                eval_msg_found = True
                                break
                        if not eval_msg_found:
                            evaluation_result_text = "*(Evaluation result prefix not found in message history)*"
                    st.markdown(evaluation_result_text)
                    st.markdown("--- End of Results ---")
                    with st.expander("Show Final Agent State (Raw)"):
                        try:
                            serializable_state = {k: (v if not isinstance(v, list) else [m.dict() if hasattr(m, 'dict') else m for m in v]) for k, v in final_state.items() if k == 'messages' or isinstance(v, (dict, str, int, float, bool, type(None)))}
                            st.json(serializable_state)
                        except Exception as json_e:
                             st.warning(f"Could not serialize final state for display: {json_e}")
                             st.write(final_state)

# Add some instructions or footer
st.divider()
st.caption("Built with Streamlit and Google Gemini. Refresh page for a new session.") 