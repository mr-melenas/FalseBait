import gradio as gr

# Gradio interface for the FastAPI application
def gradio_interface():
    def classify_url(url):
        import requests
        response = requests.post("http://127.0.0.1:8000/api/v1/predict", json={"url": url})
        return response.json()["classification"]

    interface = gr.Interface(fn=classify_url, inputs="text", outputs="text")
    interface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    gradio_interface()