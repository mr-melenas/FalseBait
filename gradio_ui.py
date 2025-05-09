import gradio as gr
import requests
from core.config import settings

def classify_url(url):
    try:
        response = requests.post(f"http://fastapi_app:8000" + settings.api_prefix + settings.api_version + "/predict", json={"url": url}, timeout=5)

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
            <p style="text-align: center; color: #9CA3AF;">
                Developed with ❤️ by group II for FactoriaF5.
            </p>
            """
        )

    demo.launch(favicon_path="images/favicon.png",server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    gradio_interface()
