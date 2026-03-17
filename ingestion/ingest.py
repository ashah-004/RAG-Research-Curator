import arxiv
import psycopg2
from datetime import datetime
import os 
import argparse

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
POSTGRES_DB = os.getenv("POSTGRES_DB", "arxiv-db")

DB_PARAMS = {
    "dbname" : POSTGRES_DB,
    "user" : POSTGRES_USER,
    "password" : POSTGRES_PASSWORD,
    "host" : POSTGRES_HOST,
    "port" : "5432"
}

def create_table_if_not_exists(cursor):
    """
    WHY: If this is the first time running the script, the table doesn't exist.
    We create it here so we don't get a "Table not found" error.
    """
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        id SERIAL PRIMARY KEY,
        arxiv_id TEXT UNIQUE,
        title TEXT,
        summary TEXT,
        published_date TIMESTAMP,
        pdf_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    print("✅ Table 'papers' is ready.")

def fetch_latest_paper(topic, limit):
    """
    WHY: We use the arxiv library to search for the specific category "cs.AI"
    (Computer Science / AI). We only fetch 1 paper for this test.
    """
    print(f"🔍 Searching ArXiv for the latest '{topic}' related papers...")
    client = arxiv.Client()
    query = f"cat:{topic}" if "cat:" not in topic and "." in topic else topic
    search = arxiv.Search(
        query = query,  # Category: Artificial Intelligence
        max_results = limit,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )

    results = list(client.results(search))
    print(f"found {len(results)} relevant papers.")

    return results

def save_paper(cursor, paper):
    """
    WHY: We insert the data. The "ON CONFLICT DO NOTHING" part is crucial.
    It means "If we already downloaded this paper ID, don't crash, just skip it."
    """
    cursor.execute("""
        INSERT INTO papers (arxiv_id, title, summary, published_date, pdf_url)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (arxiv_id) DO NOTHING;
    """, (
        paper.get_short_id(),
        paper.title,
        paper.summary,
        paper.published,
        paper.pdf_url
    ))
    print(f"💾 Saved: {paper.title}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="cs.AI", help="Arxiv category or topic")
    parser.add_argument("--limit", type=int, default=1, help="Number of papers to fetch")
    args = parser.parse_args()

    try:
        # 1. Connect to Database
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = True # Save changes immediately
        cursor = conn.cursor()

        # 2. Ensure Storage Exists
        create_table_if_not_exists(cursor)

        # 3. Get Data
        papers = fetch_latest_paper(args.topic, args.limit)

        # 4. Save Data
        for paper in papers:
            save_paper(cursor, paper)

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()
        print("🔌 Connection closed.")

if __name__ == "__main__":
    main()