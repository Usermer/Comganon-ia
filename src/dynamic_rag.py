from pathlib import Path
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from split import split_documents
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from llm import CompagnionLLM

class DynamicRAG:
    """RAG dynamique - charger un PDF à la volée"""
    
    def __init__(self):
        self.llm = CompagnionLLM(model="orca-mini")
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
    
    def load_pdf(self, pdf_path):
        """Charger et indexer un PDF"""
        try:
            # Gradio passe directement le chemin du fichier
            if isinstance(pdf_path, str):
                file_path = pdf_path
            else:
                file_path = str(pdf_path)
            
            print(f"[LOAD] Chargement du PDF: {file_path}")
            
            # Charger le PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                return None, "[X] Aucun document trouvé"
            
            print(f"[OK] {len(documents)} pages chargées")
            
            # Diviser en chunks
            chunks = split_documents(documents)
            print(f"[OK] {len(chunks)} chunks créés")
            
            # Créer l'index Chroma
            chroma_index = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name="dynamic_pdf"
            )
            
            return chroma_index, f"[✓] {len(chunks)} chunks créés à partir de {len(documents)} pages"
        
        except Exception as e:
            print(f"Erreur: {e}")
            return None, f"[X] Erreur: {str(e)}"
    
    def ask_question(self, chroma_index, question):
        """Poser une question sur le PDF"""
        results = chroma_index.similarity_search(question, k=3)
        chunks = [doc.page_content for doc in results]
        
        response = self.llm.generate_response(
            question,
            context=chunks,
            language="français"
        )
        
        return response, chunks