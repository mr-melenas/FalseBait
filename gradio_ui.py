import gradio as gr
import requests
from core.config import settings
import os

# Obtén la URL de FastAPI desde una variable de entorno, con fallback a 'http://fastapi_app:8000'
FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://fastapi_app:8000")

def classify_url(url):
    try:
        endpoint = f"{FASTAPI_BASE_URL}{settings.api_prefix}{settings.api_version}/predict"
        response = requests.post(endpoint, json={"url": url}, timeout=50)

        return response.json().get("classification", "❌ Error: Invalid output from server")
    except Exception as e:
        return f"❌ There was an error connecting to server: {e}"


def gradio_interface():
    with gr.Blocks(title=settings.proyect_name) as demo:
        gr.Markdown(
            f"""
            <div style="text-align: center;">
                <h1 style="color: #3B82F6;">🔍 {settings.proyect_name}</h1>
                <p style="font-size: 18px; color: black;">
                    {settings.description}
                </p>
            </div>
            """
        )

        url_input = gr.Textbox(
            label="🔗 Enter URL to analize",
            placeholder="https://example.com",
            show_label=True
        )

        classify_button = gr.Button("🚀 Analize")

        output = gr.Textbox(
            label="Machine Learning Classification Result",
            placeholder="Waiting for input...",
            interactive=False,
            lines=1,
        )

        classify_button.click(fn=classify_url, inputs=url_input, outputs=output)

        gr.Markdown(
            """
            <div style="display: flex; flex-direction: row; align-items: center; justify-content: center; gap: 10px; text-align: center;">
                <p style="color: #9CA3AF; margin: 0;">
                    Developed with ❤️ by group II
                </p>
                <p style="color: #FFA500; margin: 0;">
                    FactoriaF5
                </p>
            </div>
            """
        )

    demo.launch(favicon_path="images/favicon.png",server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    gradio_interface()
