import os
import requests
import tempfile
import psycopg2
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, helpers
import os 

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
POSTGRES_DB = os.getenv("POSTGRES_DB", "arxiv-db")

DB_PARAMS = {
    "dbname": POSTGRES_DB,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "host": POSTGRES_HOST,
    "port": "5432"
}

# OpenSearch Client (Local)
opensearch_client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': 9200}],
    http_auth=None,
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False
)

INDEX_NAME = "arxiv_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

def create_index_if_not_exists():
    """
    WHY: We must tell OpenSearch that the 'vector' field is special. 
    If we don't define this mapping, k-NN search will fail later.
    """
    if not opensearch_client.indices.exists(index=INDEX_NAME):
        index_body = {
            "settings": {
                "index": {
                    "knn": True  # Enable Vector Search support
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"}, # Standard Keyword Search (BM25)
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIMENSION,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib"
                        }
                    },
                    "arxiv_id": {"type": "keyword"},
                    "title": {"type": "text"}
                }
            }
        }
        opensearch_client.indices.create(index=INDEX_NAME, body=index_body)
        print(f"✅ Index '{INDEX_NAME}' created with Hybrid mappings.")
    else:
        print(f"ℹ️ Index '{INDEX_NAME}' already exists.")

def process_papers():
    # A. Connect to Postgres to get papers
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    
    # Fetch papers that have a URL (Limit 1 for testing)
    cursor.execute("SELECT arxiv_id, title, pdf_url FROM papers;")
    papers = cursor.fetchall()

    # Load Embedding Model (Downloads automatically on first run)
    print("⏳ Loading Embedding Model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    for paper in papers:
        arxiv_id, title, pdf_url = paper
        print(f"\n🚀 Processing: {title} ({arxiv_id})")

        # B. Download PDF to a temp file
        try:
            response = requests.get(pdf_url)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
        except Exception as e:
            print(f"❌ Failed to download PDF: {e}")
            continue

        # C. Load & Chunk Text
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Split: 500 chars with 50 char overlap (Preserves context)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        print(f"✂️ Split into {len(docs)} chunks.")

        # D. Embed & Prepare for OpenSearch
        actions = []
        for i, doc in enumerate(docs):
            text_content = doc.page_content
            
            # GENERATE VECTOR (The Magic Step)
            embedding = model.encode(text_content).tolist()

            # Create the JSON document for OpenSearch
            doc_body = {
                "arxiv_id": arxiv_id,
                "title": title,
                "text": text_content,       # For Keyword Search
                "vector_field": embedding,  # For Semantic Search
                "chunk_index": i
            }

            actions.append({
                "_index": INDEX_NAME,
                "_source": doc_body
            })

        # E. Bulk Upload to OpenSearch
        if actions:
            helpers.bulk(opensearch_client, actions)
            print(f"💾 Indexed {len(actions)} chunks to OpenSearch!")

        # Cleanup temp file
        os.remove(tmp_path)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    create_index_if_not_exists()
    process_papers()