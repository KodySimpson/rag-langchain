import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG Demo", page_icon="🤖", layout="wide")
st.title("RAG Question-Answering Demo")

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not set in the environment variables")
    st.stop()

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

url = st.text_input("Enter a URL to load documents from:", 
                    value="https://www.govinfo.gov/content/pkg/CDOC-110hdoc50/html/CDOC-110hdoc50.htm")

if st.button("Initialize RAG System"):
    with st.spinner("Loading and processing documents..."):
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        st.session_state.vectorstore = InMemoryVectorStore.from_documents(
            chunks, 
            embeddings
        )
        st.success("RAG system initialized successfully!")

if st.session_state.vectorstore is not None:
    llm = ChatOpenAI(model_name="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer questions about the provided documents. Use the provided context to answer the question. IMPORTANT: If you are unsure of the answer, say 'I don't know' and don't make up an answer."),
        ("user", "Question: {question}\nContext: {context}")
    ])
    
    chain = prompt | llm
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ask a Question")
        question = st.text_area("Enter your question:")
        
        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    retriever = st.session_state.vectorstore.as_retriever()
                    docs = retriever.invoke(question)
                    
                    response = chain.invoke({
                        "question": question,
                        "context": docs
                    })
                    
                    st.session_state.last_response = response.content
                    st.session_state.last_context = docs
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.subheader("Answer")
        if 'last_response' in st.session_state:
            st.write(st.session_state.last_response)
            
            with st.expander("Show Retrieved Context"):
                for i, doc in enumerate(st.session_state.last_context, 1):
                    st.markdown(f"**Relevant Document {i}:**")
                    st.markdown(doc.page_content)
                    st.markdown("---")
else:
    st.info("Please initialize the RAG system first by entering a URL and clicking 'Initialize RAG System'")