from langchain_ollama import OllamaLLM
from typing import List, Optional

class CompagnionLLM:
    """Assistant IA avec Ollama"""
    
    def __init__(self, model: str = "mistral"):
        """Initialiser le LLM"""
        try:
            self.llm = OllamaLLM(
                model=model,
                base_url="http://localhost:11434",
                temperature=0.1,  # Très bas = réponses courtes et rapides
                num_predict=200,   # Limite stricte = max 200 tokens
                top_p=0.5          # Réduit les variations = plus rapide
            )
            print(f"[OK] Modèle '{model}' connecté")
        except Exception as e:
            print(f"[X] Erreur: {e}")
            raise
    
    def generate_response(
        self,
        query: str,
        context: Optional[List[str]] = None,
        language: str = "français"
    ) -> str:
        """Générer une réponse"""
        
        if context:
            prompt = self._build_rag_prompt(query, context, language)
        else:
            prompt = self._build_simple_prompt(query, language)
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f" Erreur: {e}")
            return ""
    
    def _build_rag_prompt(self, query: str, context: List[str], language: str) -> str:
        """Construire un prompt RAG ultra-court"""
        # Prendre seulement les 2 premiers chunks pour gagner du temps
        context_text = "\n".join([chunk[:300] for chunk in context[:2]])
        
        if language.lower() == "français":
            return f"""Contexte: {context_text}

Q: {query}
R:"""
        else:
            return f"""Context: {context_text}

Q: {query}
A:"""
    
    def _build_simple_prompt(self, query: str, language: str) -> str:
        """Construire un prompt simple optimisé"""
        if language.lower() == "français":
            return f"""Réponds brièvement:

{query}"""
        else:
            return f"""Answer briefly:

{query}"""


if __name__ == "__main__":
    llm = CompagnionLLM(model="mistral")
    
    print("\n" + "="*60)
    print("TEST 1: Question simple")
    print("="*60)
    response = llm.generate_response("Qu'est-ce que les modèles de language?")
    print(f"Réponse:\n{response}\n")

