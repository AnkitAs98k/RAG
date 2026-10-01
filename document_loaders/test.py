import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import TextLoader

data =  TextLoader("rag_application_test_document.txt")

docs = data.load()
print(docs)