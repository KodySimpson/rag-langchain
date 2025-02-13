from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Load and process the document
loader = WebBaseLoader("https://www.govinfo.gov/content/pkg/CDOC-110hdoc50/html/CDOC-110hdoc50.htm")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Initialize the chat model
llm = ChatOpenAI(model_name="gpt-4")

# Create conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the conversation prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can answer questions about the US Constitution. 
                 Use the provided context to answer the question and reference previous conversation 
                 when relevant. If you are unsure of the answer, say 'I don't know'."""),
    ("system", "Previous conversation:\n{chat_history}"),
    ("user", "Question: {question}\nContext: {context}")
])

def main():
    print("Welcome to the Constitutional Q&A Assistant!")
    print("Ask questions about the US Constitution. Type 'exit' to end the conversation.\n")

    while True:
        # Get user input
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break

        # Get chat history
        chat_history = memory.load_memory_variables({})['chat_history']

        # Retrieve relevant documents
        docs = retriever.invoke(query)

        # Generate response
        response = prompt.invoke({
            "question": query,
            "context": docs,
            "chat_history": chat_history
        })
        result = llm.invoke(response)
        print(f"\nAssistant: {result.content}")

        # Save the interaction to memory
        memory.save_context(
            {"input": query},
            {"output": result.content}
        )

    # Show conversation history at the end
    print("\nFull Conversation History:")
    print(memory.load_memory_variables({})['chat_history'])

if __name__ == "__main__":
    main() 