from langchain_community.embeddings import HuggingFaceEmbeddings

print("Starting download...")
# This forces the download directly to your cache
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Download complete! You can now run Streamlit.")