import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from dynamic_rag import DynamicRAG

# =====================================================
# INITIALISATION
# =====================================================

print("⏳ Initialisation (Ollama)...")

dynamic_rag = DynamicRAG()
current_index = None

# =====================================================
# CHARGER PDF
# =====================================================

def process_pdf(pdf_file):
    global current_index

    if pdf_file is None:
        return "⚠️ Veuillez charger un PDF"

    try:
        file_path = str(pdf_file)
        current_index, msg = dynamic_rag.load_pdf(file_path)
        return f"✅ {msg}"
    except Exception as e:
        return f"❌ Erreur: {e}"

# =====================================================
# POSER UNE QUESTION ET OBTENIR LA RÉPONSE
# =====================================================

def ask_question(question):
    """Poser une question et obtenir la réponse via RAG"""
    
    if current_index is None:
        return "⚠️ Chargez un PDF d'abord"
    
    if not question.strip():
        return "⚠️ Posez une question"
    
    try:
        print(f"🔍 Recherche: {question}")
        answer, sources = dynamic_rag.ask_question(current_index, question)
        
        result = f"""
        <div style='padding:20px; font-size:15px; line-height: 1.8;'>
        
        <h3>📝 Votre Question</h3>
        <p style='background-color:#f0f0f0; padding:15px; border-radius:5px; border-left:4px solid #007bff;'>
            <b>{question}</b>
        </p>
        
        <h3>🤖 Réponse de l'IA</h3>
        <p style='background-color:#e8f4f8; padding:15px; border-radius:5px; border-left:4px solid #28a745;'>
            {answer}
        </p>
        
        <h3>📚 Source du Document</h3>
        <p style='font-size:13px; color:#666; background-color:#f9f9f9; padding:10px; border-radius:3px;'>
            <i>{sources[0][:400] if sources else 'N/A'}...</i>
        </p>
        
        </div>
        """
        
        return result
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return f"[X] Erreur: {str(e)}"

# =====================================================
# INTERFACE GRADIO - 2 PAGES SIMPLES
# =====================================================

with gr.Blocks(title="Companion IA - Questions et Réponses", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Companion IA - Questions et Réponses")
    gr.Markdown("**Étape 1:** Chargez un PDF | **Étape 2:** Posez vos questions")
    
    # PAGE 1: CHARGER PDF
    with gr.Tab("Charger le PDF"):
        gr.Markdown("### Téléchargez votre document")
        
        pdf_input = gr.File(label="Sélectionnez un PDF", file_types=[".pdf"])
        load_btn = gr.Button("Charger", variant="primary", size="lg")
        status = gr.Textbox(label="Statut", interactive=False)
        
        load_btn.click(process_pdf, pdf_input, status)
    
    # PAGE 2: POSER QUESTIONS
    with gr.Tab("Poser une Question"):
        gr.Markdown("### Posez une question sur votre document")
        
        question_input = gr.Textbox(
            label="Votre question",
            placeholder="",
            lines=3
        )
        
        ask_btn = gr.Button("Obtenir la Réponse", variant="primary", size="lg")
        
        gr.Markdown("---")
        
        result = gr.HTML(label="Résultat")
        
        ask_btn.click(ask_question, question_input, result)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("[START] Application Lancée!")
    print("="*60)
    print("[URL] Accédez à: http://localhost:7860")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        quiet=True
    )
