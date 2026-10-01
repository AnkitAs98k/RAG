from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Embeddings (must match the model used to create Chroma_db)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # 384 dimensions
    model_kwargs={"device": "cpu"}
)

# 2. Connect to existing Chroma DB
vectorStore = Chroma(
    persist_directory="Chroma_db",
    embedding_function=embeddings
)

# 3. MMR Retriever configuration
retriever = vectorStore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# 4. Groq LLM configuration
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# 5. Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)


def get_answer(query: str):
    """Processes a user question and returns the answer alongside source chunks."""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)
    return response.content, [doc.page_content for doc in docs]


# This allows running `python main.py` directly for CLI use
if __name__ == "__main__":
    print("RAG system initialized successfully.")
    print("Type '0' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query == "0":
            print("Exiting...")
            break
        
        if not query:
            continue

        answer, _ = get_answer(query)
        print(f"\nAI: {answer}\n")