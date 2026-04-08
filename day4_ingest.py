import voyageai
import chromadb
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = chromadb.CloudClient(
    api_key = os.getenv('CHROMA_API_KEY'),
    database = os.getenv('CHROMA_DATABASE'),
    tenant = os.getenv('CHROMA_TENANT')
)

vo_client = voyageai.Client()

def get_chunks(filename, chunk_size=200, overlap=20, min_chunk_size = 50):
    """
    Splits the contents of a given text file into overlapping chunks of words.

    Args:
        filename (str): Path to the text file to split.
        chunk_size (int, optional): Number of words in each chunk. Defaults to 200.
        overlap (int, optional): Number of words that overlap between consecutive chunks. Defaults to 20.

    Returns:
        list of str: List containing chunks of text, each as a string of words.
    """
    chunks = []
    with open(filename,'r', encoding='utf-8') as f:  
        f = f.read()
        sections = f.split('---') # Split by section seperator
        for section in sections:
            front_ptr = chunk_size
            back_ptr = 0  
            words = section.split()
            if len(words) < chunk_size:
                if not words:
                    continue
                chunk = ' '.join(words)
                if len(words) < min_chunk_size and chunks:
                    chunks[-1] = chunks[-1] + ' ' + chunk
                else:
                    chunks.append(chunk)
            else:
                while front_ptr <= len(words):
                    chunk_words = words[back_ptr:front_ptr]
                    while front_ptr < len(words) and  '.' not in chunk_words[-1]:
                        chunk_words.append(words[front_ptr])
                        front_ptr += 1
                    chunk = ' '.join(chunk_words)
                    chunks.append(chunk)
                    back_ptr = front_ptr - overlap
                    if front_ptr <= len(words):
                        while back_ptr > 0 and '.' not in words[back_ptr-1]:
                            back_ptr -= 1
                    front_ptr = back_ptr + chunk_size
                if (back_ptr + overlap) < len(words):
                    chunk_words = words[back_ptr:front_ptr]
                    chunk = ' '.join(chunk_words)
                    if len(chunk_words) < min_chunk_size and chunks: # Add to previous chunk if chunk too small
                        chunks[-1] = chunks[-1] + ' ' + chunk
                    else:
                        chunks.append(chunk)
    return chunks

def get_embeddings(chunks):
    """
    Generate vector embeddings for a list of text chunks using the Voyage AI API.

    Args:
        chunks (list of str): A list of textual chunks to embed.

    Returns:
        list: A list of embedding vectors (one per chunk) as returned by the embedding model.
    """
    all_embeddings = []
    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        result = vo_client.embed(batch, model='voyage-3', input_type='document').embeddings
        all_embeddings.extend(result)
        print(f'Embedded chunks {i+1}–{min(i+batch_size, len(chunks))} of {len(chunks)}')
        if i + batch_size < len(chunks):
            time.sleep(20)
    return all_embeddings

def store_chunks(chunks, embeddings, coll_name):
    """
    Stores text chunks and their corresponding embeddings into a ChromaDB collection.

    Args:
        chunks (list of str): The text chunks to store in the database.
        embeddings (list): A list of embeddings corresponding to the text chunks.
        coll_name (str): The name of the collection to create or overwrite in the database.

    Returns:
        str: A message indicating how many chunks were stored and in which collection.
    """
    if coll_name in [coll.name for coll in client.list_collections()]:
        client.delete_collection(coll_name)
    coll = client.create_collection(coll_name)
    coll.add(
        documents = chunks,
        ids = [f'doc_{i}' for i in range(len(chunks))],
        embeddings = embeddings
    )
    return f'{len(chunks)} chunks stored in {coll_name}'
    
def main():
    chunks = get_chunks('notes.txt')
    embeddings = get_embeddings(chunks)
    print(store_chunks(chunks,embeddings,'notes'))

if __name__ == '__main__':
    main()
