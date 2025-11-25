import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["TRANSFORMERS_NO_TF"] = "1"

try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

import streamlit as st
import tempfile
from pathlib import Path

from modules.pdf_processor import extract_text_from_pdf, split_into_paragraphs
from modules.keyword import load_legalbert_model, extract_legal_keywords
from modules.keyword_meaning import get_keywords_meaning_smart
from modules.vector_store import create_faiss_index
from modules.chatbot import answer_query_with_context
from modules.highlight_pdf import highlight_paragraphs_in_original_pdf
from modules.case_law_fetcher import get_cases_for_keywords
from modules.semantic_importance import analyze_paragraphs_hybrid

# --- Page Config ---
st.set_page_config(
    page_title="⚖️ Law-Lens AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Load Custom CSS ---
def load_css():
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# --- Load LegalBERT Model Once ---
@st.cache_resource
def get_kw_model(): 
    return load_legalbert_model()

kw_model = get_kw_model()

# --- Hero Section ---
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem; color: #60a5fa;'>⚖️ Law-Lens AI</h1>
        <p style='font-size: 1.2rem; color: #cbd5e1; margin-bottom: 2rem;'>
            Intelligent Legal Document Analysis Powered by AI
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Features Overview ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #1e293b; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 2px solid #334155;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🧠</div>
            <div style='font-weight: 600; color: #60a5fa;'>AI Analysis</div>
            <div style='font-size: 0.875rem; color: #cbd5e1;'>Smart keyword extraction</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #1e293b; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 2px solid #334155;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>💬</div>
            <div style='font-weight: 600; color: #60a5fa;'>Chat with Docs</div>
            <div style='font-size: 0.875rem; color: #cbd5e1;'>Ask questions instantly</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #1e293b; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 2px solid #334155;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🎨</div>
            <div style='font-weight: 600; color: #60a5fa;'>Highlighting</div>
            <div style='font-size: 0.875rem; color: #cbd5e1;'>Important clauses marked</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div style='text-align: center; padding: 1rem; background: #1e293b; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 2px solid #334155;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>⚖️</div>
            <div style='font-weight: 600; color: #60a5fa;'>Case Laws</div>
            <div style='font-size: 0.875rem; color: #cbd5e1;'>Indian court cases</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- File Upload Section ---
st.markdown("""
    <div style='background: #1e293b; padding: 2rem; border-radius: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin: 2rem 0; border: 2px solid #334155;'>
        <h3 style='color: #60a5fa; margin-bottom: 1rem;'>📄 Upload Your Legal Document</h3>
        <p style='color: #cbd5e1; margin-bottom: 1rem;'>Upload a PDF document to begin AI-powered analysis</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

# --- Processing ---
if uploaded_file:
    # Save original PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        original_pdf_path = tmp.name

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # -------- TEXT EXTRACTION --------
    @st.cache_data
    def cached_extract_text(file_bytes):
        import io
        file_like = io.BytesIO(file_bytes)
        file_like.name = uploaded_file.name
        return extract_text_from_pdf(file_like)
    
    with st.spinner("🔍 Extracting text from PDF..."):
        text = cached_extract_text(uploaded_file.getvalue())

    # -------- KEYWORD EXTRACTION --------
    with st.spinner("🧠 Identifying legal keywords with AI..."):
        keywords = extract_legal_keywords(text, kw_model, top_n=15)

    # -------- KEYWORD MEANINGS --------
    with st.spinner("💬 Generating keyword meanings..."):
        meanings = get_keywords_meaning_smart(keywords)
    
    # -------- CASE LAW FETCHING --------
    case_laws = {}
    if keywords:
        with st.spinner("⚖️ Fetching related Indian court cases..."):
            case_laws = get_cases_for_keywords(keywords[:5])

    # -------- HYBRID PARAGRAPH IMPORTANCE SCORING --------
    with st.spinner("🧠 Analyzing document importance with AI..."):
        paragraphs = split_into_paragraphs(text)
        paragraph_data = analyze_paragraphs_hybrid(paragraphs)

    # -------- CHAT INDEXING --------
    with st.spinner("💾 Preparing intelligent search index..."):
        index, chunks = create_faiss_index(text)

    # Show analysis summary
    high_count = sum(1 for p in paragraph_data if p.get("importance") == "high")
    medium_count = sum(1 for p in paragraph_data if p.get("importance") == "medium")
    low_count = sum(1 for p in paragraph_data if p.get("importance") == "low")
    
    st.success("✅ Document analyzed successfully!")
    
    # Metrics with enhanced styling
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("🔴 Critical Clauses", high_count)
    col_b.metric("🟡 Important Clauses", medium_count)
    col_c.metric("⚪ Standard Clauses", low_count)
    col_d.metric("📊 Total Paragraphs", len(paragraph_data))
    
    st.markdown("<br>", unsafe_allow_html=True)

    # -------- GENERATE HIGHLIGHTED PDF --------
    with st.spinner("📄 Creating highlighted PDF..."):
        highlighted_pdf_path = highlight_paragraphs_in_original_pdf(
            original_pdf_path,
            paragraph_data
        )

    # -------- DOWNLOAD SECTION --------
    st.markdown("""
        <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    padding: 1.5rem; border-radius: 16px; margin: 2rem 0;'>
            <h3 style='color: white; margin-bottom: 0.5rem;'>📥 Download Analyzed Document</h3>
            <p style='color: rgba(255,255,255,0.9); margin-bottom: 1rem;'>
                Get your PDF with color-coded importance highlights
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    with open(highlighted_pdf_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Highlighted PDF",
            data=f,
            file_name="Law_Lens_Analyzed.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # -------- SIDE BY SIDE: CHATBOT (LEFT) + KEYWORDS (RIGHT) --------
    col1, col2 = st.columns([1, 1])

    # -------- LEFT: CHATBOT --------
    with col1:
        st.markdown("""
            <div style='background: #1e293b; padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem; border: 2px solid #334155;'>
                <h3 style='color: #60a5fa; margin-bottom: 0.5rem;'>💬 AI Chat Assistant</h3>
                <p style='color: #cbd5e1; font-size: 0.9rem;'>Ask questions about your document</p>
            </div>
        """, unsafe_allow_html=True)

        chat_container = st.container(height=400, border=True)
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                    <div style='text-align: center; padding: 2rem; color: #64748b;'>
                        <div style='font-size: 3rem; margin-bottom: 1rem;'>💬</div>
                        <h4 style='color: #1e3a8a;'>Start a conversation</h4>
                        <p>Ask me anything about your legal document!</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        user_query = st.text_input("💭 Ask a question:", key="chat_input_box", placeholder="e.g., What are the main obligations?")

        ask_col, clear_col = st.columns([3, 1])
        ask_btn = ask_col.button("🚀 Ask AI", use_container_width=True, type="primary")
        clear_btn = clear_col.button("🗑️ Clear", use_container_width=True)

        if clear_btn:
            st.session_state.messages = []
            st.rerun()

        if ask_btn:
            if user_query.strip():
                st.session_state.messages.append({"role": "user", "content": user_query})

                with st.spinner("🤔 AI is thinking..."):
                    answer = answer_query_with_context(user_query, index, chunks)

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()
            else:
                st.warning("⚠️ Please enter a valid question.")

    # -------- RIGHT: KEYWORDS --------
    with col2:
        st.markdown("""
            <div style='background: #1e293b; padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem; border: 2px solid #334155;'>
                <h3 style='color: #60a5fa; margin-bottom: 0.5rem;'>🔑 Legal Keywords</h3>
                <p style='color: #cbd5e1; font-size: 0.9rem;'>AI-identified key legal terms</p>
            </div>
        """, unsafe_allow_html=True)

        keyword_download_list = []

        keywords_container = st.container(height=400, border=True)
        with keywords_container:
            if keywords and isinstance(meanings, dict):
                explained_count = 0

                for kw, meaning in meanings.items():
                    keyword_download_list.append(f"{kw}: {meaning}")

                    if isinstance(meaning, str) and meaning.lower() != "no explanation needed":
                        with st.expander(f"🔹 {kw}"):
                            st.write(meaning)
                        explained_count += 1

                if explained_count == 0:
                    st.info("All extracted keywords are common English words—no special legal meaning.")
            else:
                st.info("No legal keywords identified.")

        if keyword_download_list:
            st.download_button(
                label="📥 Download Keywords & Meanings",
                data="\n\n".join(keyword_download_list),
                file_name=f"{uploaded_file.name}_keywords.txt",
                mime="text/plain",
                use_container_width=True
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # -------- CASE LAWS SECTION --------
    st.markdown("""
        <div style='background: #1e293b; padding: 2rem; border-radius: 16px; margin-top: 1rem; border: 2px solid #334155;'>
            <h3 style='color: #60a5fa; margin-bottom: 1rem;'>⚖️ Related Indian Case Laws</h3>
            <p style='color: #cbd5e1;'>Relevant IPC sections and court cases for extracted keywords</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if case_laws:
        for keyword, (json_data, ui_text) in case_laws.items():
            with st.container():
                if "error" in json_data:
                    st.warning(f"⚠️ {keyword.upper()}: {json_data.get('error', 'No case found')}")
                else:
                    ipc_section = json_data.get('ipc_section', 'N/A')
                    category = json_data.get('case_category', 'N/A')
                    summary = json_data.get('summary', 'N/A')
                    kanoon_link = json_data.get('kanoon_link', '')
                    
                    st.markdown(f"""
                        <div style='background: #1e293b; padding: 1.5rem; border-radius: 12px; 
                                    border-left: 4px solid #3b82f6; margin-bottom: 1rem; 
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.3); border: 2px solid #334155;'>
                            <h4 style='color: #60a5fa; margin-bottom: 1rem;'>🔑 {keyword.upper()}</h4>
                            <p style='color: #f1f5f9;'><strong>⚖️ IPC/Law Section:</strong> {ipc_section}</p>
                            <p style='color: #f1f5f9;'><strong>📂 Category:</strong> {category.title()}</p>
                            <p style='color: #f1f5f9;'><strong>📝 Summary:</strong> {summary}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if kanoon_link:
                        st.link_button("🔗 View All Cases on IndianKanoon", kanoon_link, use_container_width=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No case laws available for the extracted keywords.")

    # -------- DOWNLOAD RAW TEXT --------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.download_button(
        label="📄 Download Extracted Text",
        data=text,
        file_name=f"{uploaded_file.name}_extracted_text.txt",
        mime="text/plain",
        use_container_width=True
    )

else:
    # Landing page when no file is uploaded
    st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; background: #1e293b; 
                    border-radius: 16px; margin: 2rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 2px solid #334155;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>📄</div>
            <h3 style='color: #60a5fa; margin-bottom: 1rem;'>Upload a Legal Document to Get Started</h3>
            <p style='color: #cbd5e1; font-size: 1.1rem;'>
                Supported format: PDF • Maximum size: 200MB
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #1e3a8a;'>✨ What Law-Lens Can Do</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background: #1e293b; padding: 2rem; border-radius: 12px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.3); border: 2px solid #334155;'>
                <h4 style='color: #60a5fa;'>🎯 Smart Analysis</h4>
                <ul style='color: #cbd5e1; line-height: 1.8;'>
                    <li>AI-powered keyword extraction using LegalBERT</li>
                    <li>Automatic importance scoring of clauses</li>
                    <li>Semantic understanding of legal concepts</li>
                    <li>Color-coded PDF highlighting</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: #1e293b; padding: 2rem; border-radius: 12px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.3); border: 2px solid #334155;'>
                <h4 style='color: #60a5fa;'>💡 Interactive Features</h4>
                <ul style='color: #cbd5e1; line-height: 1.8;'>
                    <li>Chat with your documents using AI</li>
                    <li>Find related Indian court cases instantly</li>
                    <li>Get IPC section recommendations</li>
                    <li>Export analysis and reports</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #cbd5e1; border-top: 1px solid #334155;'>
        <p style='color: #cbd5e1;'>⚖️ Law-Lens AI • Powered by LegalBERT, Groq AI & FAISS</p>
        <p style='font-size: 0.875rem; color: #94a3b8;'>Intelligent Legal Document Analysis</p>
    </div>
""", unsafe_allow_html=True)
