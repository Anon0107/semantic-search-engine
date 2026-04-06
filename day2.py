import chromadb
import requests
import voyageai
import os
from dotenv import load_dotenv

load_dotenv()

voyage_client = voyageai.Client()

client = chromadb.CloudClient(
    tenant=os.getenv('CHROMA_TENANT'),
    database=os.getenv('CHROMA_DATABASE'),
    api_key=os.getenv('CHROMA_API_KEY')
)

def get_news_id(num):
    url = 'https://hacker-news.firebaseio.com/v0/topstories.json'
    response = requests.get(url=url)
    response.raise_for_status()
    return response.json()[:num]

def get_titles(index_list):
    titles = []
    for index in index_list:
        url = f'https://hacker-news.firebaseio.com/v0/item/{index}.json'
        response = requests.get(url=url)
        response.raise_for_status()
        data = response.json()
        if data.get('title'):
            titles.append(data['title'])
    return titles

titles = get_titles(get_news_id(200))
print(f'Fetched {len(titles)} titles')

embeddings = voyage_client.embed(titles, model='voyage-3', input_type='document').embeddings
print(f'Embedded {len(embeddings)} documents')

if "hn_titles" in [c.name for c in client.list_collections()]:
    client.delete_collection("hn_titles")
collection = client.get_or_create_collection('hn_titles')

collection.add(
    documents=titles,
    ids=[f'doc_{i}' for i in range(len(titles))],
    embeddings=embeddings
)
print(f'Stored {collection.count()} documents')

query = 'artificial intelligence breakthroughs'
query_embedding = voyage_client.embed([query], model='voyage-3', input_type='query').embeddings

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)
# Semantic search can search through similar context through vector embeddings
print(f'\nTop 5 results for "{query}":')
for doc in results['documents'][0]:
    print(f'- {doc}')

# Keyword search requires exact string to be in the title
keyword = "artificial intelligence"
keyword_results = [t for t in titles if keyword.lower() in t.lower()][:5]

print(f"\nTop 5 keyword results for: '{keyword}'")
for doc in keyword_results:
    print(f"- {doc}")