import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dynamic_rag import DynamicRAG
from main import RAGSystem

# =====================================
# INITIALISATION
# =====================================

print("[INIT] Initialisation (Ollama + RAG)...")

rag_system = RAGSystem()
dynamic_rag = DynamicRAG()
current_index = None

# Initialisation du système de recommandation
print("[INIT] Chargement du dataset PEEKC...")
try:
    wiki_mapping = pd.read_csv('../dataset/PEEKC-Dataset-main/datasets/v2/id_to_wiki_metadata_mapping.csv')
    wiki_mapping['combined_text'] = wiki_mapping['title'].fillna('') + ' ' + wiki_mapping['description'].fillna('')
    wiki_mapping = wiki_mapping.dropna(subset=['title'])
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(wiki_mapping['combined_text'])
    print(f"[OK] Dataset chargé: {wiki_mapping.shape[0]} ressources (Wikipedia + YouTube + Coursera + Udemy + FreeCodeCamp)")
except Exception as e:
    print(f"[WARNING] Erreur chargement dataset: {e}")
    wiki_mapping = None
    vectorizer = None
    tfidf_matrix = None

# =====================================
# CHARGER PDF
# =====================================

def load_pdf(pdf_file):
    global current_index

    if pdf_file is None:
        return "[!] Veuillez sélectionner un PDF"

    try:
        file_path = str(pdf_file)
        current_index, _ = dynamic_rag.load_pdf(file_path)
        return "[✓] PDF chargé avec succès"
    except Exception as e:
        return f"❌ Erreur: {e}"

# =====================================
# POSER UNE QUESTION
# =====================================

def ask_question(question):
    if current_index is None:
        return "[!] Chargez d'abord un PDF", ""

    if not question.strip():
        return "[!] Entrez une question", ""

    try:
        answer, sources = dynamic_rag.ask_question(current_index, question)
        sources_text = "\n\n".join(sources[:2]) if sources else ""
        return answer, sources_text
    except Exception as e:
        return f"[X] Erreur: {e}", ""

# =====================================
# RECOMMANDATION DE MODULES
# =====================================

def recommend_modules(query, top_n=10):
    """
    Recherche les modules similaires à la requête de l'utilisateur.
    """
    if wiki_mapping is None or vectorizer is None or tfidf_matrix is None:
        return pd.DataFrame(columns=['title', 'url', 'platform'])
    
    # Vectoriser la requête
    query_vec = vectorizer.transform([query])
    
    # Calculer similarité cosinus
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Top N indices
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Résultats
    results = wiki_mapping.iloc[top_indices].copy()
    
    # Retourner sans id et similarity_score, formater URL comme lien cliquable
    return results[['title', 'url', 'platform']]

def search_modules(query, top_n=10):
    """Interface Gradio pour rechercher des modules"""
    if not query or query.strip() == "":
        return pd.DataFrame(columns=['title', 'url', 'platform'])
    
    results = recommend_modules(query, top_n=int(top_n))
    return results

# =====================================
# INTERFACE GRADIO
# =====================================

with gr.Blocks(title="Chat PDF – Ollama") as demo:
    gr.Markdown("# Chat PDF (Ollama + RAG)")

    with gr.Tab("[1] Charger PDF"):
        gr.Markdown("Sélectionnez un fichier PDF")
        pdf_file = gr.File(label="PDF", file_types=[".pdf"])
        load_btn = gr.Button("Charger le PDF", variant="primary")
        status = gr.Textbox(label="Statut", interactive=False)

        load_btn.click(load_pdf, pdf_file, status)

    with gr.Tab("[2] Poser des questions"):
        gr.Markdown("Posez des questions sur le contenu du PDF")

        question = gr.Textbox(
            label="Votre question",
            placeholder="Ex: Quel est le rôle d'un shard dans MongoDB ?",
            lines=2
        )
        ask_btn = gr.Button("Poser la question", variant="primary")

        answer = gr.Textbox(label="Réponse", lines=6, interactive=False)
        sources = gr.Textbox(label="Sources (extraits du PDF)", lines=4, interactive=False)

        ask_btn.click(ask_question, question, [answer, sources])

    with gr.Tab("[3] Recommandation de Modules"):
        gr.Markdown("# Recherche de Modules Wikipedia")
        gr.Markdown("Entrez un sujet ou module que vous cherchez (ex: machine learning, python, data science...)")
        
        with gr.Row():
            query_input = gr.Textbox(
                label="Module / Sujet recherché",
                placeholder="Tapez votre recherche ici...",
                lines=1
            )
            top_n_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Nombre de recommandations"
            )
        
        search_btn = gr.Button("Rechercher", variant="primary")
        
        results_table = gr.Dataframe(
            headers=['title', 'url', 'platform'],
            label="Résultats de recommandation",
            datatype=["str", "markdown", "str"],
            column_widths=["40%", "40%", "20%"]
        )
        
        search_btn.click(
            fn=search_modules,
            inputs=[query_input, top_n_slider],
            outputs=results_table
        )
        
        query_input.submit(
            fn=search_modules,
            inputs=[query_input, top_n_slider],
            outputs=results_table
        )

# =====================================
# LANCEMENT
# =====================================

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    show_error=True
)
