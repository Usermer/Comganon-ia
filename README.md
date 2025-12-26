# 🤖 Compagnon IA - Système RAG Éducatif

Un assistant IA intelligent basé sur **RAG (Retrieval-Augmented Generation)** qui permet d'interroger des documents PDF et de recommander des ressources d'apprentissage (Wikipedia, YouTube, Coursera, Udemy, FreeCodeCamp).

## ✨ Fonctionnalités

- 📄 **Analyse de PDF** : Chargez un document PDF et posez des questions dessus
- 🔍 **RAG Dynamique** : Recherche sémantique dans les documents avec ChromaDB
- 🤖 **IA Conversationnelle** : Génération de réponses via Ollama (Mistral/Orca-Mini)
- 📚 **Recommandation de Modules** : Suggestions de ressources d'apprentissage basées sur vos questions
- 🎨 **Interface Gradio** : Interface utilisateur intuitive et moderne

## 🚀 Technologies

- **LangChain** : Framework RAG
- **ChromaDB** : Base de données vectorielle
- **Ollama** : LLM local (Mistral, Orca-Mini)
- **Gradio** : Interface web
- **Scikit-learn** : Système de recommandation (TF-IDF + Cosine Similarity)
- **PyPDF** : Extraction de texte des PDF

## 📦 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Usermer/Comganon-ia.git
cd Comganon-ia
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Installer Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows : télécharger depuis https://ollama.com
```

### 4. Télécharger les modèles

```bash
ollama pull mistral
ollama pull orca-mini
ollama pull nomic-embed-text
```

### 5. Lancer Ollama

```bash
ollama serve
```

## 🏗️ Structure du Projet

```
Comganon-ia/
├── src/
│   ├── app.py                 # Application Gradio complète
│   ├── app_simple.py          # Version simplifiée
│   ├── main.py                # Système RAG principal
│   ├── llm.py                 # Interface LLM Ollama
│   ├── retrieve.py            # Récupération de chunks
│   ├── dynamic_rag.py         # RAG dynamique pour PDF
│   ├── embeddings_chroma.py   # Création d'index ChromaDB
│   ├── ingest.py              # Chargement de documents
│   ├── split.py               # Division en chunks
│   ├── view_chroma.py         # Visualisation de la base
│   └── search_chroma.py       # Recherche dans ChromaDB
├── docs/                      # Documents PDF à indexer
├── data/
│   └── chroma_db/            # Base de données vectorielle
├── dataset/
│   └── PEEKC-Dataset-main/   # Dataset de recommandations
└── requirements.txt

```

## 🎯 Utilisation

### Mode 1 : Interface Gradio (Recommandé)

```bash
cd src
python app.py
```

Accédez à l'interface sur `http://localhost:7860`

### Mode 2 : Interface Simple

```bash
cd src
python app_simple.py
```

### Mode 3 : Ligne de commande

```bash
cd src
python main.py
```

## 🔧 Configuration

### Changer le modèle LLM

Dans `src/llm.py` :

```python
self.llm = OllamaLLM(
    model="mistral",  # ou "orca-mini", "llama2", etc.
    temperature=0.1,
    num_predict=200
)
```

### Ajuster les paramètres RAG

Dans `src/split.py` :

```python
splitter = CharacterTextSplitter(
    chunk_size=1000,      # Taille des chunks
    chunk_overlap=200     # Chevauchement
)
```

## 📚 Créer votre propre index

1. **Placer vos PDF** dans le dossier `docs/`

2. **Créer l'index ChromaDB** :

```bash
cd src
python embeddings_chroma.py
```

3. **Vérifier l'index** :

```bash
python view_chroma.py
```

4. **Tester la recherche** :

```bash
python search_chroma.py
```

## 🎓 Dataset de Recommandations

Le système utilise le **PEEKC Dataset** avec plus de 30 000 ressources :
- Wikipedia
- YouTube
- Coursera
- Udemy
- FreeCodeCamp

## 🐛 Résolution de Problèmes

### Ollama ne répond pas

```bash
# Vérifier qu'Ollama tourne
ollama list

# Relancer le serveur
ollama serve
```

### Index ChromaDB vide

```bash
# Recréer l'index
cd src
python embeddings_chroma.py
```

### Erreur de modèle manquant

```bash
# Télécharger le modèle
ollama pull nomic-embed-text
```

## 📊 Performances

| Opération | Temps moyen |
|-----------|-------------|
| Chargement PDF | ~2-5s |
| Recherche ChromaDB | ~0.5s |
| Génération réponse | ~3-10s |
| Recommandations | ~0.2s |

## 🤝 Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à :
- 🐛 Signaler des bugs
- 💡 Proposer de nouvelles fonctionnalités
- 📝 Améliorer la documentation

## 📄 Licence

MIT License - Voir le fichier LICENSE pour plus de détails

## 🙏 Remerciements

- [LangChain](https://langchain.com)
- [Ollama](https://ollama.com)
- [ChromaDB](https://www.trychroma.com)
- [Gradio](https://gradio.app)
- [PEEKC Dataset](https://github.com/PEEKC/PEEKC-Dataset)

---

Développé avec ❤️ par [Usermer](https://github.com/Usermer)