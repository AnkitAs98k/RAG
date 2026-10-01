
from dotenv import load_dotenv

# Document loading & splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# Embeddings & Vector Store
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables (OPENAI_API_KEY, GROQ_API_KEY, etc.)
load_dotenv()

# 1. Load data (raw string avoids Windows backslash escape issues)
loader = PyPDFLoader("document_loaders\GRU.pdf")
docs = loader.load()

# 2. Split data into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)

# 3. Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"}
)

# 4. Store embeddings in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./Chroma_db"
)

print(f"Successfully processed {len(chunks)} chunks and persisted to ./Chroma_db")