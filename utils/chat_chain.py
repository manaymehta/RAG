from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import os

def get_chain(vectorstore):
    llm = ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def get_answer(chain, question):
    result = chain.invoke({"question": question})
    sources = []
    for doc in result.get("source_documents", []):
        sources.append(doc.page_content[:200].replace("\n", " ") + "...")
    return result["answer"], sources
