import voyageai
import chromadb
from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

client = chromadb.CloudClient(
    api_key = os.getenv('CHROMA_API_KEY'),
    database = os.getenv('CHROMA_DATABASE'),
    tenant = os.getenv('CHROMA_TENANT')
)

vo_client = voyageai.Client()

anthropic_client = anthropic.Anthropic()

def get_rephrase(query):
    """
    Uses the Anthropic Claude model to rephrase a user's question.

    Args:
        query (str): The original user question to be rephrased.

    Returns:
        str: The rephrased version of the provided question.
    """
    response = anthropic_client.messages.create(
        model = 'claude-haiku-4-5-20251001',
        max_tokens = 1024,
        system = "You are a helpful assistant. Rephrase user's question and respond with ONLY the rephrased question.",
        messages = [{'role': 'user', 'content': query}]
    )
    return next((b.text for b in response.content if b.type == 'text'),query)

def get_embeddings(query):
    """
    Generate and return the embedding vector for a given query string 
    using the VoyageAI 'voyage-3' model.

    Args:
        query (str): The input query to embed.

    Returns:
        list: The embedding vector for the query.
    """
    embeddings = vo_client.embed([query], model = 'voyage-3', input_type = 'query').embeddings
    return embeddings

def get_documents(query, coll_name):
    """
    Retrieves the top 3 most relevant documents from the specified ChromaDB collection 
    for a given query embedding.

    Args:
        query (list): The embedding vector(s) representing the query.
        coll_name (str): The name of the ChromaDB collection to search.

    Returns:
        list or None: List of the top 3 document strings if available, 
                     or None if the collection does not exist.
    """
    if coll_name not in [coll.name for coll in client.list_collections()]:
        return None
    coll = client.get_collection(coll_name)
    results = coll.query(
        query_embeddings = query,
        n_results = 3
    )
    return results['documents'][0]

def get_prompt(query, documents):
    """
    Constructs a prompt for the language model using the provided query and source documents.

    Args:
        query (str): The question or query to be answered.
        documents (list): A list of relevant context documents (strings) to include in the prompt.

    Returns:
        str: The formatted prompt string including the context and question.
    """
    sources = ''
    for index, doc in enumerate(documents,1):
        sources += f'Chunk {index}:\n{doc}\n\n-----\n\n'

    prompt = f"""<context>{sources}</context>
    <question>{query}</question>"""
    return prompt
    
def get_answer(prompt):
    """
    Sends a prompt to the language model and retrieves the model's textual response.

    Args:
        prompt (str): The formatted input prompt containing context and query.

    Returns:
        str: The textual answer provided by the model. Returns 'No responses' if no textual content is found.
    """
    response = anthropic_client.messages.create(
        model = 'claude-haiku-4-5-20251001',
        max_tokens = 1024,
        system = "You are a helpful assistant. Answer ONLY using the context given. DO NOT assume anything, always flag uncertainty when answering. Respond I don't know if answer not in context. State which context chunk used for response.",
        messages = [{'role': 'user', 'content': prompt}]
    )
    return next((b.text for b in response.content if b.type == 'text'),'No responses')

def main():
    coll_name = 'notes'
    query = input('User: ').strip()
    query = get_rephrase(query)
    query_embeddings = get_embeddings(query)
    documents = get_documents(query_embeddings, coll_name)
    prompt = get_prompt(query, documents)
    result = get_answer(prompt)
    print('\nClaude: ', end = '')
    print(result)
    print('\nSources:\n')
    for i in range(len(documents)):
        print(f'Chunk {i+1}:\n{documents[i]}\n')

if __name__ == '__main__':
    main()
