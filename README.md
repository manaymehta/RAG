# 📚 RAG-based Chatbot with Vector Visualization

This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to:

- Upload a `.md` document  
- Visualize embedded document chunks in 3D using **t-SNE**
- Cluster document chunks with **K-Means** for insight
- Chat with the content using **Groq LLM** + **LangChain**
- All powered via an interactive **Streamlit** web interface

---

## 🧠 Features

- **File Upload**: Upload your own `.md` file  
- **Vectorization**: Uses `nomic-ai/nomic-embed-text-v1` for text embeddings  
- **Vector DB**: Powered by `Chroma` for fast retrieval  
- **Visualization**: 3D interactive `plotly` graph with t-SNE + KMeans clustering  
- **Conversational RAG**: Chat with file contents via `meta-llama/llama-4-scout-17b-16e-instruct` (Groq)  
- **Memory**: Tracks conversation context using LangChain’s memory  
- **Live Interface**: Clean UI with Streamlit  

---

## 🚀 Online - https://testvectorrag.streamlit.app/ 

---

## 🧪 Tech Stack

- **Streamlit** – Web UI  
- **LangChain** – RAG + Memory  
- **ChromaDB** – Vector store  
- **HuggingFace Embeddings** – Text embeddings  
- **Groq LLM API** – Fast inference  
- **Plotly + scikit-learn** – t-SNE & clustering (Visualization)