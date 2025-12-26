from pathlib import Path
import chromadb

# Configuration
project_root = Path(__file__).parent.parent
persist_dir = project_root / "data" / "chroma_db"

print("\n" + "="*70)
print("VISUALISATION DE LA BASE CHROMA")
print("="*70 + "\n")

# Vérifier que la base existe
if not Path(persist_dir).exists():
    print(f"[X] La base Chroma n'existe pas à : {persist_dir}")
    exit(1)

print(f"[DB] Chemin de la base : {persist_dir}\n")

# Se connecter à la base existante directement
try:
    # Utiliser ChromaDB directement sans passer par LangChain
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(name="rag_documents")
    
    # Récupérer la collection
    collection = client.get_collection(name="rag_documents")
    
    # Obtenir les statistiques
    count = collection.count()
    print(f"[INFO] Nombre de documents dans Chroma : {count}\n")
    
    if count == 0:
        print("La base est vide.")
        exit(0)
    
    # Afficher les documents
    print("="*70)
    print("CONTENU DE LA BASE CHROMA")
    print("="*70 + "\n")
    
    # Récupérer tous les documents
    all_data = collection.get(include=["documents", "metadatas", "embeddings"])
    
    for i, (doc_id, document, metadata) in enumerate(
        zip(all_data["ids"], all_data["documents"], all_data["metadatas"]), 1
    ):
        print(f"\n[DOC] Document {i}/{count}")
        print(f"   ID: {doc_id}")
        print(f"   Source: {metadata.get('source', 'N/A')}")
        print(f"   Page: {metadata.get('page', 'N/A')}")
        preview = document[:200].replace('\n', ' ')
        print(f"   Contenu: {preview}...")
        print(f"   Taille: {len(document)} caractères")
    
    print("\n" + "="*70)
    print(f"✅ Total: {count} documents dans la base")
    print("="*70 + "\n")
    
    # Option : Afficher la taille de la base
    if persist_dir.exists():
        db_size = sum(f.stat().st_size for f in persist_dir.rglob('*') if f.is_file())
        print(f"💾 Taille de la base : {db_size / 1024 / 1024:.2f} MB\n")

except Exception as e:
    print(f"[X] Erreur : {e}")
    print("\nAssurez-vous que:")
    print("  1. Ollama est en train de tourner (ollama serve)")
    print("  2. Le modèle nomic-embed-text est téléchargé (ollama pull nomic-embed-text)")
    print("  3. La base Chroma existe et contient des données")
