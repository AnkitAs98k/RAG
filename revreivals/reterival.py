from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_groq import ChatGroq
load_dotenv()

embeddings = HuggingFaceEmbeddings()


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]


vectorStore =  Chroma.from_documents(docs ,  embeddings)





#similarity search code implemetnation

similarity_retrivers = vectorStore.as_retriever(
     search_type="similarity", 
     search_kwargs={"k": 3}
)

similarity_docs = similarity_retrivers.invoke("what is Gradient descent?")

for docs in similarity_docs:
    print(docs.page_content)
    
    
    
    
    
    
    
    

# this is the code for MMR 

mmr_retrivers = vectorStore.as_retriever(
     search_type="mmr", 
     search_kwargs={"k": 3, "lambda_mult": 0.5}
)

mmr_docs = mmr_retrivers.invoke("what is Gradient descent?")

for docs in similarity_docs:
    print(docs.page_content)
    
    
    
    
    
    
    
    
# this is the code for multiquery 


mmr_retrivers = vectorStore.as_retriever(
     search_type="mmr", 
     search_kwargs={"k": 3, "lambda_mult": 0.5}
)

mmr_docs = mmr_retrivers.invoke("what is Gradient descent?")

for docs in similarity_docs:
    print(docs.page_content)
    


#this is the code for Multiquery

retreivers =  vectorStore.as_retriever()
model = ChatGroq(model = "openai/gpt-oss-120b")


multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retreivers,
    llm=model
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)


print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)