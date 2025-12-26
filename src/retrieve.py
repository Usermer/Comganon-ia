from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from typing import List


def load_chroma_index():
    db_path = Path(__file__).parent.parent / "data" / "chroma_db"

    if not db_path.exists():
        raise FileNotFoundError(f"Index non trouvé: {db_path}")
    
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    chroma_index = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
        collection_name="rag_documents"
    )
    return chroma_index


def retrieve_chunks(query:str,chroma_index,top_k:int=5)->List[str]:
    results=chroma_index.similarity_search(query,k=top_k)
    chunks=[doc.page_content for doc in results]
    return chunks


if __name__=="__main__":
    print("chargement de l'index chroma ..")
    chroma_index=load_chroma_index()
    print("index chargé\n")

    query="Language Models"
    print(f"recherche :'{query}'")
    chunks=retrieve_chunks(query,chroma_index,top_k=3)

    print(f"{len(chunks)} resultats: \n")
    for i,chunk in enumerate (chunks,1):
        preview=chunk[:100].replace('\n',' ')
        print(f"{i}. {preview}\n")

