from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
import os

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")

# --- CONFIGURATION ---
OPENSEARCH_CONFIG = {
    'hosts': [{'host': OPENSEARCH_HOST, 'port': 9200}],
    'http_auth': None,
    'use_ssl': False,
    'verify_certs': False,
    'ssl_show_warn': False
}

INDEX_NAME = "arxiv_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

print("🚀 Script is starting...")

class SearchEngine:
    def __init__(self):
        print("⏳ Loading Embedding Model for Search...")
        # We load the SAME model used in ingestion. 
        # If we used a different model, the vectors wouldn't match!
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        self.client = OpenSearch(**OPENSEARCH_CONFIG)
        print("✅ Search Engine Ready.")
    
    def search(self, query, k=3):
        """
        Performs Hybrid Search (Vector + Keyword) with RRF Reranking
        """
        # 1. Vector Search Query
        query_vector = self.model.encode(query).tolist()
        vector_query = {
            "knn": {
                "vector_field": {
                    "vector": query_vector,
                    "k": k  # Get top k vector matches
                }
            }
        }

        # 2. Keyword Search Query (BM25)
        keyword_query = {
            "match": {
                "text": query
            }
        }

        # 3. Hybrid Query (OpenSearch Pipeline)
        # Note: Standard OpenSearch needs a specific query structure for hybrid.
        # For simplicity without plugins, we will run TWO queries and merge them manually (RRF).
        
        # Run Vector Search
        print(f"🔍 Running Vector Search...")
        vec_response = self.client.search(
            index=INDEX_NAME, 
            body={"size": k, "query": vector_query, "_source": ["text", "title"]}
        )
        
        # Run Keyword Search
        print(f"🔍 Running Keyword Search...")
        key_response = self.client.search(
            index=INDEX_NAME, 
            body={"size": k, "query": keyword_query, "_source": ["text", "title"]}
        )

        # 4. RRF Algorithm (Reciprocal Rank Fusion)
        # Score = 1 / (rank + 60)
        hits_map = {}

        # Process Vector Hits
        for rank, hit in enumerate(vec_response['hits']['hits']):
            doc_id = hit['_id']
            if doc_id not in hits_map:
                hits_map[doc_id] = {'text': hit['_source']['text'], 'score': 0}
            hits_map[doc_id]['score'] += 1.0 / (rank + 60)

        # Process Keyword Hits
        for rank, hit in enumerate(key_response['hits']['hits']):
            doc_id = hit['_id']
            if doc_id not in hits_map:
                hits_map[doc_id] = {'text': hit['_source']['text'], 'score': 0}
            hits_map[doc_id]['score'] += 1.0 / (rank + 60)

        # Sort by final RRF score
        sorted_hits = sorted(hits_map.values(), key=lambda x: x['score'], reverse=True)
        
        # Return top k fused results
        print(f"📊 Merged {len(sorted_hits)} results using RRF.")
        return [item['text'] for item in sorted_hits[:k]]

if __name__ == "__main__":
    # This block only runs if you execute this file directly
    engine = SearchEngine()
    
    # Try a question relevant to the paper you downloaded in Step 2!
    # (Since I don't know exactly which paper you got, try a generic AI term)
    engine.search("What is multigraph based agentic memorty architecture for AI Agents?")