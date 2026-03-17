import gradio as gr 
import requests 
import json
import os 

API_HOST = os.getenv("API_HOST", "localhost")

API_URL = f"http://{API_HOST}:8000/chat/stream"
INGEST_URL = f"http://{API_HOST}:8000/ingest"

def query_api(message, history, k_val):
    """
    Sends the user's message to our FastAPI backend and streams the result.
    """
    if history is None:
        history = []

    history.append([message, ""])
    try:
        #  payload for api
        payload = {"query": message, "k": int(k_val)}

        #  stream connection
        with requests.post(API_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                history[-1][1] = f"Error: {response.text}"
                yield history
                return 

            partial_text = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')

                    if decoded_line.startswith("data: "):
                        token = decoded_line.replace("data: ", "")
                        partial_text += token

                        history[-1][1] = partial_text
                        yield history
    except Exception as e:
        history[-1][1] =  f"Connection Error: Is the API running? ({e})"
        yield history

def trigger_ingestion(topic, limit):
    try:
        payload = {"topic": topic, "limit": int(limit)}
        requests.post(INGEST_URL, json=payload)
        return f"Started! check your Docker terminal logs for progress"
    except Exception as e:
        return f"Error: {e}" 

with gr.Blocks(title="Arxiv RAG") as demo:
    gr.Markdown("# Arxiv research Assistant")

    with gr.Tabs():
        with gr.TabItem("Chat"):
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(height=400)
                    msg = gr.Textbox(label="Question")
                with gr.Column(scale=1):
                    k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="context chunks (k)")
            clear = gr.ClearButton([msg, chatbot])
            msg.submit(query_api, [msg, chatbot, k_slider], [chatbot])

        with gr.TabItem("New Papers"):
            gr.Markdown("Download Papers by Topic")

            with gr.Row():
                topic_box = gr.Textbox(value="cs.AI", label="Arxiv Category/Topic")
                limit_box = gr.Slider(minimum=1, maximum=20, value=3, step=1, label="Numbers of Papers")

            ingest_btn = gr.Button("Start Downloading", variant="primary")
            status_out = gr.Textbox(label="Status", interactive=False)

            ingest_btn.click(trigger_ingestion, [topic_box, limit_box], status_out)
 

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)