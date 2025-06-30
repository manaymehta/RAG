import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from utils.visualizer import plot_tsne
from utils.chat_chain import get_chain, get_answer
from utils.embed_store import create_vectorstore
 
# Load environment variables
load_dotenv()

st.set_page_config(layout="wide", page_title="RAG Explorer")
st.title("📄 RAG Explorer: Chat with a Markdown File")

# Upload a Markdown file
st.sidebar.markdown("### Step 1: Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a Markdown file (.md)", type=["md"])

# Caching vectorstore
@st.cache_resource
def cached_vectorstore(filepath):
    return create_vectorstore(filepath)
 
# Set session defaults
if "vectorstore" not in st.session_state: 
    st.session_state.vectorstore = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_types" not in st.session_state:
    st.session_state.doc_types = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Process uploaded file only once
if uploaded_file and st.session_state.vectorstore is None:
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Processing file and generating embeddings..."):
            vectorstore, chunks, doc_types = cached_vectorstore(filepath)
            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks
            st.session_state.doc_types = doc_types
            st.session_state.conversation_chain = get_chain(vectorstore)

    st.success("✅ File embedded and stored!")
    st.info(f"📄 Document split into **{len(st.session_state.chunks)}** chunks.")


# Visualization
if st.session_state.vectorstore:
    st.subheader("📊 Vector Visualization (t-SNE)")
    fig = plot_tsne(st.session_state.vectorstore, st.session_state.doc_types)
    st.plotly_chart(fig, use_container_width=True)

# Chat interface
if st.session_state.conversation_chain:
    st.subheader("💬 Ask Questions")
    user_input = st.chat_input("Ask a question...")

    if user_input:
        with st.spinner("Generating answer..."):
            answer, sources = get_answer(st.session_state.conversation_chain, user_input)
            st.session_state.chat_history.append((user_input, answer, sources))

    for user_msg, response, source_chunks in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Bot:** {response}")
        if source_chunks:
            with st.expander("📄 Source(s)"):
                for s in source_chunks:
                    st.markdown(f"> {s}")

# Reset button
if st.sidebar.button("🔄 Reset App"):
    st.session_state.clear()
    st.experimental_rerun()
