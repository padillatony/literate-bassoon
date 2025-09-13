
import gradio as gr
from backend.model import SavantModel

savant = SavantModel("./savant_finetuned")  # usa el modelo entrenado
try:
    savant.load()
except:
    print("⚠️ No se encontró modelo entrenado, usando modelo base.")
    savant = SavantModel("distilgpt2")
    savant.load()

def responder(mensaje, historial):
    respuesta = savant.infer(mensaje)
    historial = historial + [(mensaje, respuesta)]
    return historial, historial

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 SAVANT-RRF ChatBox")

    chatbot_ui = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Escribe aquí", placeholder="Pregunta algo...")
    enviar = gr.Button("🚀 Enviar")
    clear = gr.Button("🧹 Limpiar Chat")

    enviar.click(responder, [msg, chatbot_ui], [chatbot_ui, chatbot_ui])
    msg.submit(responder, [msg, chatbot_ui], [chatbot_ui, chatbot_ui])
    clear.click(lambda: [], None, chatbot_ui, queue=False)

demo.launch(share=True)
