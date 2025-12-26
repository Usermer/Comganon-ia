import time
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from ingest import load_documents_from_folder
from split import split_documents

print("\n" + "="*70)
print("CRÉATION DE L'INDEX CHROMA")
print("="*70 + "\n")

project_root = Path(__file__).parent.parent
docs_path = project_root / "docs"
persist_dir = project_root / "data" / "chroma_db"

# Créer le dossier si nécessaire
persist_dir.mkdir(parents=True, exist_ok=True)

# ÉTAPE 1: Charger les documents
print(" Chargement des documents...")
start = time.time()
documents = load_documents_from_folder(docs_path)
load_time = time.time() - start
print(f"{len(documents)} documents en {load_time:.2f} secondes\n")

# ÉTAPE 2: Diviser en chunks
print(" Division en chunks...")
start = time.time()
chunks = split_documents(documents)
split_time = time.time() - start
print(f" {len(chunks)} chunks en {split_time:.1f} secondes\n")

# ÉTAPE 3: Créer les embeddings
print(" Création des embeddings Ollama...")
print("Peut prendre du temps (surtout si c'est la première fois)...")

try:
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    # Test de connexion
    test_embedding = embeddings.embed_query("test")
    print(f" Ollama connecté (embedding de dimension {len(test_embedding)})\n")
except Exception as e:
    print(f" Erreur de connexion à Ollama: {e}")
    print("Vérifiez que Ollama est installé et lancé :")
    print("  - Installation : curl -fsSL https://ollama.com/install.sh | sh")
    print("  - Lancer : ollama serve")
    print("  - Télécharger le modèle : ollama pull nomic-embed-text")
    exit(1)

# ÉTAPE 4: Créer l'index Chroma
print(" Création de l'index Chroma...")
start = time.time()

try:
    # Préparer les données
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    # Ajouter un ID unique pour chaque chunk
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Créer l'index avec persistance AUTOMATIQUE
    # La persistance se fait automatiquement quand on spécifie persist_directory
    chroma_index = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=str(persist_dir),
        collection_name="rag_documents"
    )
    
 
    embeddings_time = time.time() - start
    print(f"Index créé avec succès en {embeddings_time:.1f} secondes\n")
    
    # Vérifier que l'index a été sauvegardé
    if persist_dir.exists() and any(persist_dir.iterdir()):
        print(f" Index sauvegardé dans : {persist_dir}")
    else:
        print(" L'index n'a pas été sauvegardé, vérifiez les permissions")
    
except Exception as e:
    print(f" Erreur lors de la création de l'index: {e}")
    exit(1)

# ÉTAPE 5: Test de recherche
print(" Test de recherche...")
test_queries = [
    "machine learning",
    "intelligence artificielle",
    "apprentissage automatique"
]

try:
    for query in test_queries:
        print(f"\n Requête : '{query}'")
        results = chroma_index.similarity_search(query, k=2)
        print(f" {len(results)} résultats trouvés")
        
        for i, doc in enumerate(results, 1):
            preview = doc.page_content[:150].replace('\n', ' ')
            source = doc.metadata.get('source', 'Inconnu')
            page = doc.metadata.get('page', 'N/A')
            print(f"  {i}. [{source} - page {page}]")
            print(f"     {preview}...\n")
    
except Exception as e:
    print(f"  Erreur lors de la recherche: {e}")
    print("Mais l'index a probablement été créé avec succès")

# Résumé final
print("\n" + "="*70)
print(" SUCCÈS !")
print("="*70)

total_time = load_time + split_time + embeddings_time
print(f"⏱  Temps total : {total_time:.1f} secondes")
print(f" Statistiques :")
print(f"   - Documents chargés : {len(documents)}")
print(f"   - Chunks créés : {len(chunks)}")
print(f"   - Embedding modèle : nomic-embed-text")
print(f"   - Index sauvegardé : {persist_dir}")
print(f"   - Collection : rag_documents")

# Vérification du stockage
print(f"\n Vérification du stockage :")
if persist_dir.exists():
    db_size = sum(f.stat().st_size for f in persist_dir.rglob('*') if f.is_file())
    print(f"   - Taille de la base : {db_size / 1024 / 1024:.2f} MB")
    num_files = len(list(persist_dir.rglob('*')))
    print(f"   - Nombre de fichiers : {num_files}")
else:
    print("    Le dossier de persistance n'existe pas")

print("="*70)

