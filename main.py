import streamlit as st
import asyncio
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
import tempfile
import os
import zipfile
import json
import requests
from bs4 import BeautifulSoup
import markdown
from urllib.parse import urlparse


# Initialize session state first
def initialize_session_state():
    """Initialize all required session state variables"""
    defaults = {
        "authenticated": False,
        "current_user": None,
        "user_email": None,
        "show_login": False,
        "show_signup": False,
        "show_forgot_password": False,
        "chat_history": [],
        "vector_store": None,
        "processed_files": set(),
        "show_landing": True,
        "pdf_enabled": True,
        "notion_enabled": False,
        "wiki_enabled": False,
        "show_sources": True,
        "uploaded_content": []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize session state
initialize_session_state()

# =========================
# 🔹 API Key
# =========================
GOOGLE_API_KEY = "AIzaSyASFxwnbFkW1lkrl9bBkZIy2xZ0CDy9MSY"

# =========================
# 🔹 Page Config & Styling
# =========================
st.set_page_config(
    page_title="IQBot-An Intelligent Q&A Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(145deg, #0f172a 0%, #1e293b 50%, #111827 100%) !important;
            color: #ffffff !important;
        }
        .user-bubble {
            background: linear-gradient(135deg, #2563eb, #1e40af);
            color: white;
            padding: 12px 18px;
            border-radius: 20px;
            margin: 8px 0;
            max-width: 70%;
            align-self: flex-end;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        }
        .bot-bubble {
            background: linear-gradient(135deg, #f97316, #ea580c);
            color: white;
            padding: 12px 18px;
            border-radius: 20px;
            margin: 8px 0;
            max-width: 70%;
            align-self: flex-start;
            box-shadow: 0 4px 15px rgba(249, 115, 22, 0.3);
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            padding: 1rem;
        }
        .sidebar-section {
            padding: 1rem 0;
            border-bottom: 1px solid #444;
        }
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #ffffff !important;
        }
        .content-item {
            background: rgba(255, 255, 255, 0.08);
            padding: 8px;
            margin: 4px 0;
            border-radius: 6px;
            font-size: 0.9rem;
            color: #ffffff !important;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
            background: rgba(255, 255, 255, 0.05);
            color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# 🔹 Event loop fix
# =========================
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# =========================
# 🔹 Content Processing Functions
# =========================
def process_pdf_file(file):
    """Process uploaded PDF file and return text chunks"""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            return None, "No text could be extracted from the PDF."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = splitter.split_text(text)
        return chunks, None

    except Exception as e:
        return None, f"Error processing PDF: {str(e)}"


def process_notion_export(file):
    """Process Notion export zip file"""
    try:
        chunks = []
        with zipfile.ZipFile(file, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.md'):
                    with zip_ref.open(file_name) as md_file:
                        content = md_file.read().decode('utf-8')
                        # Convert markdown to plain text
                        html = markdown.markdown(content)
                        soup = BeautifulSoup(html, 'html.parser')
                        text = soup.get_text()

                        if text.strip():
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=100,
                                length_function=len
                            )
                            file_chunks = splitter.split_text(text)
                            chunks.extend(file_chunks)

        return chunks, None if chunks else "No readable content found in Notion export."

    except Exception as e:
        return None, f"Error processing Notion export: {str(e)}"


def process_wiki_url(url):
    """Process Wikipedia or other wiki URL"""
    try:
        headers = {'User-Agent': 'NoteBot/1.0 (Educational Use)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract main content
        content_selectors = ['#mw-content-text', '.mw-parser-output', 'main', 'article']
        text = ""

        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                text = content_div.get_text(separator=' ', strip=True)
                break

        if not text:
            text = soup.get_text(separator=' ', strip=True)

        if not text.strip():
            return None, "No readable content found at the URL."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = splitter.split_text(text)
        return chunks, None

    except Exception as e:
        return None, f"Error processing URL: {str(e)}"


# =========================
# 🔹 Vector Store Functions
# =========================
def create_vector_store(chunks):
    """Create FAISS vector store from text chunks"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store, None
    except Exception as e:
        return None, f"Error creating vector store: {str(e)}"


def update_vector_store(new_chunks):
    """Update existing vector store with new chunks"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        if st.session_state.vector_store is None:
            st.session_state.vector_store = FAISS.from_texts(new_chunks, embeddings)
        else:
            new_vector_store = FAISS.from_texts(new_chunks, embeddings)
            st.session_state.vector_store.merge_from(new_vector_store)

        return True, None
    except Exception as e:
        return False, f"Error updating vector store: {str(e)}"


# =========================
# 🔹 Answer Generation
# =========================
def get_answer_simple(user_query, vector_store):
    """Get answer using simple approach"""
    try:
        matching_chunks = vector_store.similarity_search(user_query, k=5)

        if not matching_chunks:
            return "I don't know Manavendra", []

        context = "\n\n".join([chunk.page_content for chunk in matching_chunks])

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            max_output_tokens=1024,
            google_api_key=GOOGLE_API_KEY
        )

        prompt = f"""You are my assistant tutor.
        Answer the question based on the following context.
        If you cannot answer based on the context, simply say "I don't know Manavendra".

        Context: {context}
        Question: {user_query}

        Answer:"""

        response = llm.invoke(prompt)
        output = response.content if hasattr(response, 'content') else str(response)

        sources = [{
            "content": chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content,
        } for chunk in matching_chunks]

        return output, sources

    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}", []


# =========================
# 🔹 Enhanced Sidebar
# =========================
def render_enhanced_sidebar():
    """Render the enhanced sidebar with multiple upload options"""
    with st.sidebar:
        st.markdown("### 📁 Upload Content")

        # PDF Files Section
        with st.expander("📄 PDF Files", expanded=st.session_state.pdf_enabled):
            st.session_state.pdf_enabled = st.toggle("Enable PDF upload", value=st.session_state.pdf_enabled)

            if st.session_state.pdf_enabled:
                st.markdown("""
                <div class="upload-area">
                    <p>📄 Drag and drop files here</p>
                    <p><small>Limit 200MB per file • PDF</small></p>
                </div>
                """, unsafe_allow_html=True)

                pdf_files = st.file_uploader(
                    "Browse files",
                    type="pdf",
                    accept_multiple_files=True,
                    key="pdf_uploader",
                    label_visibility="collapsed"
                )

                if pdf_files:
                    for pdf_file in pdf_files:
                        if pdf_file.name not in [item['name'] for item in st.session_state.uploaded_content if
                                                 item['type'] == 'PDF']:
                            with st.spinner(f"Processing {pdf_file.name}..."):
                                chunks, error = process_pdf_file(pdf_file)
                                if chunks:
                                    success, vector_error = update_vector_store(chunks)
                                    if success:
                                        st.session_state.uploaded_content.append({
                                            'name': pdf_file.name,
                                            'type': 'PDF',
                                            'chunks': len(chunks)
                                        })
                                        st.success(f"✅ {pdf_file.name} processed!")
                                    else:
                                        st.error(f"Vector store error: {vector_error}")
                                else:
                                    st.error(f"Error: {error}")

        # Notion Exports Section
        with st.expander("📝 Notion Exports", expanded=st.session_state.notion_enabled):
            st.session_state.notion_enabled = st.toggle("Enable Notion exports", value=st.session_state.notion_enabled)

            if st.session_state.notion_enabled:
                notion_file = st.file_uploader(
                    "Upload Notion export (ZIP file)",
                    type="zip",
                    key="notion_uploader"
                )

                if notion_file:
                    if notion_file.name not in [item['name'] for item in st.session_state.uploaded_content if
                                                item['type'] == 'Notion']:
                        with st.spinner(f"Processing {notion_file.name}..."):
                            chunks, error = process_notion_export(notion_file)
                            if chunks:
                                success, vector_error = update_vector_store(chunks)
                                if success:
                                    st.session_state.uploaded_content.append({
                                        'name': notion_file.name,
                                        'type': 'Notion',
                                        'chunks': len(chunks)
                                    })
                                    st.success(f"✅ {notion_file.name} processed!")
                                else:
                                    st.error(f"Vector store error: {vector_error}")
                            else:
                                st.error(f"Error: {error}")

        # Wiki Pages Section
        with st.expander("🌐 Wiki Pages", expanded=st.session_state.wiki_enabled):
            st.session_state.wiki_enabled = st.toggle("Enable Wiki pages", value=st.session_state.wiki_enabled)

            if st.session_state.wiki_enabled:
                wiki_url = st.text_input("Enter Wikipedia or wiki URL:", key="wiki_url")

                if st.button("Add Wiki Page", key="add_wiki"):
                    if wiki_url:
                        parsed_url = urlparse(wiki_url)
                        page_name = parsed_url.path.split('/')[-1] or parsed_url.netloc

                        if page_name not in [item['name'] for item in st.session_state.uploaded_content if
                                             item['type'] == 'Wiki']:
                            with st.spinner(f"Processing {page_name}..."):
                                chunks, error = process_wiki_url(wiki_url)
                                if chunks:
                                    success, vector_error = update_vector_store(chunks)
                                    if success:
                                        st.session_state.uploaded_content.append({
                                            'name': page_name,
                                            'type': 'Wiki',
                                            'chunks': len(chunks),
                                            'url': wiki_url
                                        })
                                        st.success(f"✅ {page_name} processed!")
                                        st.session_state.wiki_url = ""  # Clear input
                                    else:
                                        st.error(f"Vector store error: {vector_error}")
                                else:
                                    st.error(f"Error: {error}")
                        else:
                            st.warning("This page has already been added!")
                    else:
                        st.warning("Please enter a valid URL!")

        # Show Sources Toggle
        st.markdown("---")
        st.session_state.show_sources = st.checkbox("🔍 Show Sources", value=st.session_state.show_sources)

        # Uploaded Content Summary with Debug Info
        if st.session_state.uploaded_content:
            st.markdown("---")
            st.markdown("### 📚 Uploaded Content")
            for item in st.session_state.uploaded_content:
                icon = {"PDF": "📄", "Notion": "📝", "Wiki": "🌐", "HTML": "🌐"}.get(item['type'], "📄")
                st.markdown(f"""
                <div class="content-item">
                    {icon} <strong>{item['name']}</strong><br>
                    <small>{item['type']} • {item['chunks']} chunks</small>
                </div>
                """, unsafe_allow_html=True)

            # Debug: Show vector store info
            if st.session_state.vector_store:
                st.markdown("*Debug Info:*")
                st.write(f"Vector store has {st.session_state.vector_store.index.ntotal} total vectors")

                # Test a sample chunk
                if st.button("🔍 Test Metadata"):
                    test_results = st.session_state.vector_store.similarity_search("test", k=1)
                    if test_results:
                        chunk = test_results[0]
                        st.write(f"Sample chunk metadata: {getattr(chunk, 'metadata', 'No metadata found')}")

        # Control Buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        with col2:
            if st.button("🗑 Clear All", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.vector_store = None
                st.session_state.uploaded_content = []
                st.success("All content cleared!")
                st.rerun()

        if st.button("🏠 Back to Landing", use_container_width=True):
            st.session_state.show_landing = True
            st.rerun()


# =========================
# 🔹 Landing Page
# =========================
import streamlit as st

def show_landing_page():
    # Wide layout for full-width boxes
    st.set_page_config(
        page_title="IQBot - Intelligent Q&A Bot",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS: Full-page and card styling
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            html, body, [data-testid="stAppViewContainer"] {
                font-family: 'Inter', sans-serif !important;
                background: linear-gradient(145deg, #0f172a 0%, #1e293b 50%, #111827 100%) !important;
                margin:0 !important;
                padding:0 !important;
                min-height:100vh !important;
            }
            .main-container {
                background: none;
                padding: 0;
                border-radius: 0;
                box-shadow: none;
                max-width: 1600px;
                margin: auto;
            }
            .gradient-text {
                background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
            }
            .feature-card {
                width: 100%;
                max-width: 1160px;
                margin: 1.8rem auto;
                background: rgba(255,255,255,0.05);
                padding: 2rem 2rem 1.5rem 2rem;
                border-radius: 18px;
                border-left: 6px solid #60a5fa;
                box-shadow: 0 12px 40px rgba(0,0,0,0.2);
                transition: transform 0.22s, box-shadow 0.22s;
            }
            .feature-card:hover {
                transform: translateY(-4px) scale(1.012);
                box-shadow: 0 10px 24px rgba(100, 30, 220, 0.33);
            }
            .feature-card h3 {
                color: #93c5fd;
                margin-bottom: 0.5rem;
                font-size: 1.45rem;
            }
            .feature-card p {
                color: #e2e8f0;
                font-size: 1.05rem;
            }
            .center-button {
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 3rem 0;
            }
            .stButton > button {
                width: 320px;
                height: 75px;
                font-size: 1.5rem;
                font-weight: 600;
                color: white !important;
                background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
                border: none;
                border-radius: 16px;
                box-shadow: 0 6px 14px rgba(37,99,235,0.29);
                transition: all 0.3s;
            }
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 26px rgba(124,58,237,0.30);
                background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
            }
            .footer {
                text-align: center;
                margin-top: 3.2rem;
                color: #94a3b8;
                font-size: 0.92rem;
                opacity: 0.8;
            }
        </style>
    """, unsafe_allow_html=True)

    # Main heading and subheading
    st.markdown("""
        <div class="main-container">
            <h1 style="text-align: center; font-size: 3rem; margin-bottom: 1.2rem;" class="gradient-text">
                📘 IQBot - Intelligent Q&A Assistant
            </h1>
            <p style="text-align: center; font-size: 1.35rem; margin-bottom: 2rem; color: #e2e8f0;">
                Transform your documents into conversational knowledge.
            </p>
            <p style="text-align: center; font-size: 1.12rem; margin-bottom: 2.4rem; color: #cbd5e1;">
                Chat with PDFs, Notion exports, and Wiki pages using AI-powered intelligence.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Features - each as a full-width card
    st.markdown("<h2 style='text-align: center; color: #f1f5f9; margin-top: 2.5rem;'>✨ Key Features</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class="feature-card">
            <h3>📄 Document Intelligence</h3>
            <p>Extract insights from PDF documents with AI-powered analysis.</p>
        </div>
        <div class="feature-card">
            <h3>🔍 Advanced Search</h3>
            <p>Find answers quickly with our powerful semantic search technology.</p>
        </div>
        <div class="feature-card">
            <h3>🗂 Multi-Format Support</h3>
            <p>Works with PDFs, Notion exports, Wiki pages, and more.</p>
        </div>
        <div class="feature-card">
            <h3>💬 Natural Conversations</h3>
            <p>Interact with your documents through intuitive, human-like dialogue.</p>
        </div>
    """, unsafe_allow_html=True)

    # Centered CTA Button
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    if st.button("🚀 Start Now", key="start-now-main-button"):
        st.session_state.show_landing = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


    # Footer
    st.markdown("""
        <div class="footer">
            <p>@made by Manavendra</p>
        </div>
    """, unsafe_allow_html=True)

# Call the function (or integrate into your app's navigation)
# show_landing_page()

# =========================
# 🔹 Main Application
# =========================
def main_app():
    """Main application"""
    st.header("📘IQBot-Intelligent Q&A Assistant ")

    # Render the enhanced sidebar
    render_enhanced_sidebar()

    # Main chat interface
    if st.session_state.uploaded_content and st.session_state.vector_store:
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                st.markdown(
                    f"<div class='chat-container'><div class='user-bubble'>{chat['content']}</div></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-container'><div class='bot-bubble'>{chat['content']}</div></div>",
                    unsafe_allow_html=True
                )

                # Show sources for historical messages if available and enabled
                if st.session_state.show_sources and 'sources' in chat and chat[
                    'sources'] and "I don't know Manavendra" not in chat['content']:
                    with st.expander(f"📚 Sources for message {(i // 2) + 1}", expanded=False):
                        for j, source in enumerate(chat['sources']):
                            # Display source metadata header
                            metadata = source.get('metadata', {})
                            source_file = metadata.get('source_file', 'Unknown')
                            page_number = metadata.get('page_number', 'N/A')
                            chunk_id = metadata.get('chunk_id', f'{j + 1}')

                            # Create header with file and page info
                            if page_number != 'N/A':
                                header = f"📄 *{source_file}* - Page {page_number}"
                            else:
                                header = f"📄 *{source_file}*"

                            st.markdown(header)
                            st.text_area(
                                f"source_{i}_{j}",
                                value=source['content'],
                                height=120,
                                disabled=True,
                                label_visibility="collapsed"
                            )

                            # Add URL if available (for wiki sources)
                            if 'url' in metadata:
                                st.markdown(f"🔗 [View Original]({metadata['url']})")

                            st.markdown("---")

        # User query input
        user_query = st.chat_input("💬 Ask a question about your uploaded content...")

        if user_query:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            with st.spinner("🤔 Thinking..."):
                output, sources = get_answer_simple(user_query, st.session_state.vector_store)

            # Add bot response to chat history with sources
            bot_message = {"role": "bot", "content": output}
            if sources:
                bot_message["sources"] = sources
            st.session_state.chat_history.append(bot_message)

            st.rerun()

    elif not st.session_state.uploaded_content:
        st.info("👈 Please upload content using the sidebar to start chatting with your documents.")
    else:
        st.error("There was an issue processing your content. Please try uploading again.")


# =========================
# 🔹 Main App Flow
# =========================
if st.session_state.show_landing:
    show_landing_page()
else:
    main_app()