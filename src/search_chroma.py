from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration
project_root = Path(__file__).parent.parent
persist_dir = project_root / "data" / "chroma_db"

print("\n" + "="*70)
print("RECHERCHE DANS LA BASE CHROMA")
print("="*70 + "\n")

# Se connecter à la base
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

chroma_index = Chroma(
    persist_directory=str(persist_dir),
    embedding_function=embeddings,
    collection_name="rag_documents"
)

# Faire une recherche
query = input("Entrez votre requête : ")

print(f"\n[SEARCH] Recherche pour : '{query}'\n")

results = chroma_index.similarity_search(query, k=5)

print(f"[OK] {len(results)} résultats trouvés:\n")

for i, result in enumerate(results, 1):
    print(f"{'='*70}")
    print(f"Résultat {i}")
    print(f"{'='*70}")
    print(f"Source: {result.metadata.get('source', 'N/A')}")
    print(f"Page: {result.metadata.get('page', 'N/A')}")
    print(f"\nContenu:\n{result.page_content}\n")
