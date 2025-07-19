import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Load the .env file
load_dotenv()

# Load data
loader = CSVLoader("tamilnadu_crime_dataset.csv")
documents = loader.load()

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Save embeddings to FAISS vector store
db = FAISS.from_documents(docs, embedding)
db.save_local("crime_faiss_index")

print("✅ Embedding complete and FAISS index saved.")
