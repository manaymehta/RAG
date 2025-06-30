from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_vectorstore(filepath):
    loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()

    # Add artificial doc_type metadata
    for doc in documents:
        doc.metadata['doc_type'] = 'user_uploaded'

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
    separators=["\n\n", "\n", " ", ""]
)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True}
    )

    vectorstore = Chroma.from_documents(chunks, embeddings)

    doc_types = ["user_uploaded"]

    return vectorstore, chunks, doc_types
