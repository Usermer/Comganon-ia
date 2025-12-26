# Compagnon IA (Local) — Chat PDF (RAG) + Recommandation de Modules

Application locale avec interface **Gradio** :
- **Chat PDF (RAG)** via **Ollama + LangChain + Chroma**
- **Recommandation de ressources** (Wikipedia + YouTube + Coursera + Udemy + FreeCodeCamp) via **TF‑IDF + cosine similarity**

## Ce repo n’inclut pas les datasets
Le dossier `dataset/` (PEEKC et autres) est **exclu** via `.gitignore`.

## Prérequis
- Python 3.12+
- Ollama installé et lancé (serveur par défaut sur `http://localhost:11434`)
- Git

## Installation
```powershell
cd "c:\Users\Admin\Desktop\ai projects\compagon_ia_v2"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset (PEEKC)
L’app lit par défaut :
`dataset/PEEKC-Dataset-main/datasets/v2/id_to_wiki_metadata_mapping.csv`

Option 1 (recommandée) : cloner/télécharger PEEKC dans `dataset/PEEKC-Dataset-main`.

## Lancer l’application
```powershell
cd "c:\Users\Admin\Desktop\ai projects\compagon_ia_v2\src"
python app.py
```
Puis ouvrir : `http://127.0.0.1:7860`

## Notes
- L’index Chroma local est dans `data/chroma_db/` (exclu du repo).
- Les fichiers PDF sont exclus du repo (`*.pdf`).
