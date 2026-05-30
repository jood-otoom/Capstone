# app/services/rag_service.py

import os
import glob
import re
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.vectorstore_path = settings.VECTORSTORE_PATH
        self.docs_dir = settings.DOCS_DIR
        
    # def _semantic_chunking(self, raw_text, source_name):
    #     """Splits text by 'Article' and injects metadata tags directly into the chunk."""
    #     # Split by the word "Article" or Arabic "مادة"
    #     raw_chunks = re.split(r'(?i)(?=Article\s\d+|مادة\s\d+)', raw_text)
        
    #     documents = []
    #     for chunk in raw_chunks:
    #         if len(chunk.strip()) > 20: # Ignore empty splits
    #             # Metadata Injection Strategy: Wrap the chunk in context
    #             enriched_text = f"[Context: Jordanian Traffic Law | Source: {source_name}]\n{chunk.strip()}"
    #             documents.append(Document(page_content=enriched_text))
                
    #     return documents

    def _semantic_chunking(self, raw_text, source_name):
        """Splits text exactly by the numeric section headers (e.g., ١. , ١٫١. ) found in the PSD Manual."""
        
        # Regex Explanation:
        # \n          : Looks for a new line
        # [\d٠-٩]+    : Matches standard digits OR Arabic-Indic digits
        # (?:[٫\.][\d٠-٩]+)? : Optionally matches the decimal/sub-section (like .1 or ٫١)
        # \.          : Matches the literal period after the number
        # \s+         : Matches the space before the title
        
        raw_chunks = re.split(r'(?=\n[\d٠-٩]+(?:[٫\.][\d٠-٩]+)?\.\s+)', raw_text)
        
        documents = []
        for chunk in raw_chunks:
            chunk_text = chunk.strip()
            
            # Ignore tiny fragments (like empty pages or random line breaks)
            if len(chunk_text) > 50: 
                # Inject the mandatory metadata tag
                enriched_text = f"[Context: Jordanian Traffic Law | Source: {source_name}]\n{chunk_text}"
                documents.append(Document(page_content=enriched_text))
                
        return documents

    def ingest_pdfs(self):
        """Scans the docs folder, parses tables, applies semantic chunking, and saves to FAISS."""
        print(f"Scanning for PDFs in {self.docs_dir}/...")
        
        # This is the line that was missing! It finds all PDFs in the directory.
        pdf_files = glob.glob(os.path.join(self.docs_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"Error: No PDFs found in {self.docs_dir}/")
            return

        all_docs = []
        for pdf_path in pdf_files:
            print(f"Parsing and chunking: {pdf_path}...")
            loader = PDFPlumberLoader(pdf_path)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            
            filename = os.path.basename(pdf_path)
            all_docs.extend(self._semantic_chunking(full_text, filename))
            
        print("Creating Unified Multilingual Vector Store locally...")
        vectorstore = FAISS.from_documents(all_docs, self.embeddings)
        vectorstore.save_local(self.vectorstore_path)
        print(f"Successfully saved FAISS index to {self.vectorstore_path}")

    def get_relevant_law(self, query="right of way intersection roundabout rear-end collision rules"):
        """Retrieves relevant law excerpts for the agent based on the crash type."""
        if not os.path.exists(self.vectorstore_path):
            return "[System Warning: Vector database not built. Run ingestion first.]"
            
        vectorstore = FAISS.load_local(
            self.vectorstore_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Pull top 5 most relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
        docs = retriever.invoke(query)
        
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    rag = RAGService()
    rag.ingest_pdfs()    
