import anthropic
import voyageai
import chromadb
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
anthropic_client = anthropic.Anthropic()
vo_client = voyageai.Client()
chroma_client = chromadb.CloudClient(
    api_key = os.getenv('CHROMA_API_KEY'),
    database = os.getenv('CHROMA_DATABASE'),
    tenant = os.getenv('CHROMA_TENANT')
)

def get_chunks(filename, chunk_size=200, overlap=20, min_chunk_size = 50):
    """
    Split a UTF-8 text file into overlapping, sentence-aligned chunks.

    The file is first split into sections by `---`, then each section is chunked
    by words while trying to end/start chunk boundaries on sentence breaks ('.').
    Very small trailing chunks are merged into the previous chunk.
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
                chunks.append(' '.join(words))
            else:
                while front_ptr <= len(words):
                    while front_ptr <= len(words) and  '.' not in words[front_ptr-1]:
                        front_ptr += 1
                    chunks.append(' '.join(words[back_ptr:front_ptr]))
                    back_ptr = front_ptr - overlap
                    if front_ptr <= len(words):
                        while back_ptr > 0 and '.' not in words[back_ptr-1]:
                            back_ptr -= 1
                    front_ptr = back_ptr + chunk_size
                if (back_ptr + overlap) < len(words):
                    if (len(words) - back_ptr) < min_chunk_size and chunks: # Add to previous chunk if chunk too small
                        chunks[-1] = chunks[-1] + ' ' + ' '.join(words[back_ptr:front_ptr])
                    else:
                        chunks.append(' '.join(words[back_ptr:front_ptr]))
    return chunks

def get_embeddings(chunks):
    """
    Generate document embeddings for text chunks in batched API calls.

    Uses the Voyage `voyage-3` model with `document` input type and throttles
    requests between batches to reduce rate-limit issues.
    """
    all_embeddings = []
    batch_size = 10
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
    Store chunk text and embeddings in a ChromaDB collection.

    Each chunk is assigned a timestamp-based unique id before insertion.
    """
    coll = chroma_client.get_or_create_collection(coll_name)
    coll.add(
        documents = chunks,
        ids = [f"{datetime.now().strftime('%Y%m%d%H%M%S')}_doc_{i}" for i in range(len(chunks))],
        embeddings = embeddings
    )
    time.sleep(1)
    return f'{len(chunks)} chunks stored in {coll_name}'
    
def ingest(filename, coll_name):
    """
    End-to-end ingestion pipeline for a source file into a collection.

    Reads/chunks the file, embeds each chunk, stores the results, and prints
    a short status message.
    """
    chunks = get_chunks(filename)
    embeddings = get_embeddings(chunks)
    print(store_chunks(chunks,embeddings,coll_name))
    return

def get_rephrase(query):
    """
    Rephrase a user query using Anthropic and return the rewritten question.

    Falls back to the original query when no text block is returned.
    """
    response = anthropic_client.messages.create(
        model = 'claude-haiku-4-5-20251001',
        max_tokens = 1024,
        system = "You are a helpful assistant. Rephrase user's question and respond with ONLY the rephrased question.",
        messages = [{'role': 'user', 'content': query}]
    )
    return next((b.text for b in response.content if b.type == 'text'),query)

def get_query_embeddings(query):
    """Generate and return a query embedding vector using Voyage `voyage-3`."""
    embeddings = vo_client.embed([query], model = 'voyage-3', input_type = 'query').embeddings
    return embeddings

def get_documents(query, coll_name):
    """Retrieve the top 3 most relevant documents from a Chroma collection."""
    coll = chroma_client.get_collection(coll_name)
    results = coll.query(
        query_embeddings = query,
        n_results = 3
    )
    return results['documents'][0]

def get_prompt(query, documents):
    """
    Build an XML-like prompt containing retrieved chunks and the user question.

    Documents are labeled as `Chunk N` and separated for readability.
    """
    sources = ''
    for index, doc in enumerate(documents,1):
        sources += f'Chunk {index}:\n{doc}\n\n-----\n\n'

    prompt = f"""<context>{sources}</context>
    <question>{query}</question>"""
    return prompt
    
def get_message(messages):
    """
    Request a non-streaming Anthropic response for a chat history.

    Returns the text reply plus input/output token usage for cost tracking.
    """
    response = anthropic_client.messages.create(
                model = 'claude-haiku-4-5-20251001',
                system = 'You are a CLI chatbot that respond gracefully',
                max_tokens = 1024,
                messages = messages
            )
    
    input_token = response.usage.input_tokens
    output_token = response.usage.output_tokens
    result = next((b.text for b in response.content if b.type == 'text'),'No responses')
    return result,input_token,output_token

def stream_message(messages):
    """
    Stream an Anthropic response to stdout and return final text with token usage.

    Appends a truncation note when generation stops due to max token limit.
    """
    print(f'Claude: ',end = '',flush = True)
    with anthropic_client.messages.stream(
                model = 'claude-haiku-4-5-20251001',
                system = "You are a CLI chatbot that respond gracfully. Answer ONLY using the context given, previous context given and summary context. DO NOT assume anything, always flag uncertainty when answering. Respond I don't know if answer not in context.",
                max_tokens = 1024,
                messages = messages
            ) as stream:
        for text in stream.text_stream:
            print(text, end ='', flush = True)
        print()
        message = stream.get_final_message()
        text = stream.get_final_text()

    input_token = message.usage.input_tokens
    output_token = message.usage.output_tokens
    if message.stop_reason == 'max_tokens':
        text = f'{text} (Max tokens reached, response truncated)'
    return text,input_token,output_token

def count_tokens(messages):
    """Estimate input token count for a message list using Anthropic's tokenizer."""
    count = anthropic_client.messages.count_tokens(
                model = 'claude-haiku-4-5-20251001',
                system ='You are a CLI chatbot that respond gracefully',
                messages = messages
    )
    return count.input_tokens

def main():
    coll_name = 'CLI_chatbot_coll'
    # Option to ingest
    while True:
        try:
            filename = input('Enter filename to ingest (Press Ctrl + C to skip): ')
            ingest(filename, coll_name)
            break
        except KeyboardInterrupt:
            print('Skipping ingest...')
            break 
        except FileNotFoundError:
            print("File not found, make sure it's uploaded")
            return
        except Exception as e:
            print(f'Unexpected error: {e}')
            continue
    if coll_name not in [c.name for c in chroma_client.list_collections()]:
        print('Collection empty, please ingest a file')
        return
    total = 0
    sec_exp = 0
    max_retry = 5
    messages = []
    print('"Press ctrl+c or type quit to exit, type /help for list of commands"')
    while True:
        if sec_exp == 0:
            try:
                message = input('User: ')
                if message.startswith('/'):
                    if message == '/help':
                        print('Commands: /help /clear /cost /clearcoll')
                    elif message == '/clear':
                        messages = []
                    elif message == '/cost':
                        print(f'Total cost: ${total:.6f}')
                    elif message == '/cleardocuments':
                        chroma_client.delete_collection(coll_name)
                        print(f'Collection {coll_name} cleared')
                        break
                    else:
                        print(f'Unknown command {message}, type /help for list of commands')
                    continue
                if message.strip() == 'quit':
                    break
                if not message.strip():
                    continue
                # Construct prompt
                message = get_rephrase(message)
                embeddings = get_query_embeddings(message)
                chunks = get_documents(embeddings,coll_name)
                prompt = get_prompt(message,chunks)
                message = {'role':'user', 'content':prompt}
            except KeyboardInterrupt:
                break
        messages.append(message)  
        # Error handling
        try:
            if count_tokens(messages) > 40000: #Token limit for conversation history
                # Compress history with a one line summary to limit token usage
                messages.pop()
                messages.append({'role':'user','content':'Summarize this conversation into a single concise message.Only respond with the message'})
                summary, input_tokens, output_tokens = get_message(messages)
                print('Compressing chat history...')
                cost = (input_tokens/1000000) * 1.00 + (output_tokens/1000000) * 5.00 # Token costs for Claude-Haiku
                total += cost
                print(f"\nInput_tokens: {input_tokens}, Output_tokens: {output_tokens}, Estimated cost: ${cost:.6f}, Total cost: ${total:.6f}\n")
                message = {'role':'user','content':f'Context summary: {summary}. {prompt}'}
                messages = [message]
            reply, input_tokens, output_tokens = stream_message(messages)
        except anthropic.RateLimitError:
            if sec_exp >= max_retry:
                print('Max retries reached, exiting')
                break
            wait = min(2**sec_exp, 60) # Cap at 60s
            print(f'Too many requests, retrying in {wait}s')
            time.sleep(wait)
            sec_exp += 1
            messages.pop()
            continue
        except anthropic.AuthenticationError:
            print('Invalid or missing API key, check .env file')
            break
        except anthropic.InternalServerError:
            if sec_exp >= max_retry:
                print('Max retries reached, exiting')
                break
            wait = min(2**sec_exp, 60) # Cap at 60s
            print(f'Server too busy, retrying again in {wait}s')
            time.sleep(wait)
            sec_exp += 1
            messages.pop()
            continue
        except anthropic.APIConnectionError:
            print('Bad connection, try again')
            sec_exp = 0
            messages.pop()
            continue
        except Exception as e:
            print(f'Unexpected error: {e}')
            break
        sec_exp = 0
        cost = (input_tokens/1000000) * 1.00 + (output_tokens/1000000) * 5.00 # Token costs for Claude-Haiku
        total += cost
        print(f"\nInput_tokens: {input_tokens}, Output_tokens: {output_tokens}, Estimated cost: ${cost:.6f}, Total cost: ${total:.6f}\n")
        print('Sources:\n')
        for i in range(len(chunks)):
            print(f'Chunk {i+1}:\n{chunks[i]}\n')
        messages.append({'role':'assistant', 'content':reply}) # Add response to chat history to preserve context
        embeddings = get_embeddings([reply]) # Add response to chromadb to add historical conversation context
        store_chunks([reply],embeddings,coll_name)

    print('Exiting...')

if __name__ == '__main__':
    main()