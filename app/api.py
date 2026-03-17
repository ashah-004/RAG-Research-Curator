import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.core.search_logic import SearchEngine
import os 
import subprocess
import threading

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")

app = FastAPI(title="Arxiv RAG API")

# configuration for our main ollama model
OLLAMA_URL = f"http://{OLLAMA_HOST}:11434/api/generate"
MODEL_NAME = "llama3.2"

print("Booting up the Search Engine...")
retriever = SearchEngine()
print("API Ready...")

# schema of request
class ChatRequest(BaseModel):
    query: str
    k: int = 3

# schema for ingest request we make
class IngestRequest(BaseModel):
    topic: str
    limit: int

def stream_processor(query: str, k: int):
    # asking the search engine for top k chunks
    relevant_chunks = retriever.search(query, k=k)

    # for no matches found i.e. relevant_chunks is empty.
    if not relevant_chunks:
        yield "data: I couldn't find any relevant documents. \n\n"
        return

    context_text = "\n\n".join(relevant_chunks)

    prompt = f"""
    You are a helpful research assistant. Use the following context to answer the user's question.
    If the answer is not in the context, say "I don't know based on the provided documents."    
    
    Context: {context_text}
    
    Question: {query}
    
    Answer:
    """

    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": True}

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    # process line
                    json_response = json.loads(line)
                    token = json_response.get("response", "")

                    if token:
                        yield f"data: {token}\n\n"
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"

def run_ingestion_scripts(topic: str, limit: int):
    """
    Runs the existing ingestion scripts as subprocesses.
    """
    try:
        print(f"🚀 API Triggering Ingestion: {topic} ({limit} papers)")
        
        # 1. Run ingest.py with arguments
        # We assume the container's working directory is /app
        cmd1 = ["python", "ingestion/ingest.py", "--topic", topic, "--limit", str(limit)]
        subprocess.run(cmd1, check=True)
        
        # 2. Run process_and_index.py
        cmd2 = ["python", "ingestion/process_and_index.py"]
        subprocess.run(cmd2, check=True)
        
        print("✅ Ingestion Sequence Complete")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Script failed: {e}")

@app.post("/ingest")
async def trigger_ingestion(request: IngestRequest):
    task = threading.Thread(target=run_ingestion_scripts, args=(request.topic, request.limit))
    task.start()
    return {"status": "started", "message": f"Ingesting {request.limit} papers onn {request.topic}"}
    

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    The endpoint your frontend will talk to.
    """
    return StreamingResponse(
        stream_processor(request.query, request.k),
        media_type="text/event-stream"
    )

@app.get('/health')
def health_check():
    return {"status": "ok", "model": MODEL_NAME}




    

