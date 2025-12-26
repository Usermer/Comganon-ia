import time
from retrieve import load_chroma_index, retrieve_chunks
from llm import CompagnionLLM

class RAGSystem:
    """Système RAG complet"""
    
    def __init__(self):
        """Initialiser le système"""
        print("\n" + "="*70)
        print(" INITIALISATION DU SYSTÈME RAG")
        print("="*70 + "\n")
        
        print(" Chargement de l'index Chroma...")
        try:
            self.chroma_index = load_chroma_index()
            print(" Index chargé\n")
        except Exception as e:
            print(f" Erreur: {e}")
            raise
        
        print(" Initialisation du LLM...")
        try:
            self.llm = CompagnionLLM(model="orca-mini")
            print(" LLM prêt\n")
        except Exception as e:
            print(f" Erreur: {e}")
            raise
        
        print("="*70)
        print(" SYSTÈME PRÊT")
        print("="*70 + "\n")
    
    def answer_question(self, question: str, top_k: int = 3) -> dict:
        """Répondre à une question"""
        start_time = time.time()
        
        print("="*70)
        print(f" Question: {question}")
        print("="*70 + "\n")
        
        # Récupérer les chunks
        print(" Recherche des documents...")
        ret_start = time.time()
        chunks = retrieve_chunks(question, self.chroma_index, top_k=top_k)
        ret_time = time.time() - ret_start
        print(f" {len(chunks)} documents en {ret_time:.2f}s\n")
        
        # Générer la réponse
        print(" Génération de la réponse...")
        gen_start = time.time()
        response = self.llm.generate_response(
            question, 
            context=chunks, 
            language="français"
        )
        gen_time = time.time() - gen_start
        print(f" Réponse générée en {gen_time:.2f}s\n")
        
        # Afficher
        print("─"*70)
        print(" RÉPONSE:")
        print("─"*70)
        print(response)
        print("\n" + "─"*70)
        print(" SOURCES:")
        print("─"*70)
        for i, chunk in enumerate(chunks, 1):
            preview = chunk[:80].replace('\n', ' ')
            print(f"{i}. {preview}...")
        print("─"*70)
        
        total_time = time.time() - start_time
        print(f"  Temps: {total_time:.2f}s")
        print("="*70 + "\n")
        
        return {
            "question": question,
            "answer": response,
            "sources": chunks,
            "time": total_time
        }


if __name__ == "__main__":
    # Initialiser
    rag = RAGSystem()
    
    # Poser des questions
    questions = [
        "Qu'est-ce que le machine learning?",
        "Comment fonctionne le deep learning?",
    ]
    
    for question in questions:
        result = rag.answer_question(question, top_k=3)

