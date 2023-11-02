from flask import Flask, request
import json
import os
import time
import requests
import queue
import threading
from synology import OutgoingWebhook
from settings import *
from chat_prompts import *
from memory import LoadCoreMemories, VECTORSTORE, VECTORSTORE2, InMemoryEntityStore, ENTITY_STORE
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.memory import VectorStoreRetrieverMemory, ConversationBufferWindowMemory, CombinedMemory, ReadOnlySharedMemory, ConversationEntityMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.retrievers import WikipediaRetriever

app = Flask(__name__)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

chat_mode=CHAT_MODE

task_queue = queue.Queue()
processing_semaphore = threading.Semaphore(value=1)

llm = LlamaCpp(
    model_path=f"./model/{MODEL_FILENAME}",
    n_ctx=CONTEXT_LENGTH,
    n_gpu_layers=GPU_LAYERS,
    max_tokens=MAX_TOKENS,
    temperature=TEMPURATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    stop=STOP_WORDS,
    repeat_penalty=REPEAT_PENALTY,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True
    )

#Load Vectorstore knowledge and memories
LoadCoreMemories()

#rag mode memory
rag_retriever = VECTORSTORE2.as_retriever(search_type="mmr", search_kwargs={'k': 3})
knowledge_mem = VectorStoreRetrieverMemory(exclude_input_keys=[], input_key="input", memory_key="knowledge", retriever=rag_retriever)
readonlymemory = ReadOnlySharedMemory(memory=knowledge_mem)

#chat mode memory
chat_retriever = VECTORSTORE.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, 'k': 1})
long_memory = VectorStoreRetrieverMemory(exclude_input_keys=["history", "entities"], input_key="input", memory_key="chat_history", retriever=chat_retriever)
entity_memory = ConversationEntityMemory(llm=llm, input_key="input", entity_store=ENTITY_STORE)
memory = CombinedMemory(memories=[long_memory, entity_memory])

#wiki mode search
wiki_retriever = WikipediaRetriever(load_max_docs=1)

#Custom Prompts
rag_prompt = PromptTemplate(
    template=RAG_TEMPLATE,
    input_variables=["input", "knowledge"],
)
chat_prompt = PromptTemplate(
    template=CHAT_TEMPLATE,
    input_variables=["input", "history", "chat_history", "entities"],
)

#Langchain chains
rag_chain = LLMChain(llm=llm, prompt=rag_prompt, memory=readonlymemory, verbose=True)
chat_chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory, verbose=True)
wiki_chain = ConversationalRetrievalChain.from_llm(llm, retriever=wiki_retriever, verbose=True)

#sending response back to synology chat
def send_back_message(user_id, output_text):
    response = output_text
    chunks = []
    current_chunk = ""
    sentences = response.split("\n\n")
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 2 <= 256:
            current_chunk += sentence + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    for chunk in chunks:
        payload = 'payload=' + json.dumps({
            'text': chunk,
            "user_ids": [int(user_id)]
        })
        try:
            response = requests.post(INCOMING_WEBHOOK_URL, payload)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return "Error", 500
        finally:
            processing_semaphore.release()
    return "success"

#message generating code
def initialize_model():
    global chat_mode
    warmup_input = "Human: Name the planets in the solar system? Assistant:"
    llm(warmup_input, max_tokens=50)
    print(f"Llama is Loaded in {chat_mode} mode")

def set_chat_mode(mode, user_id):
    global chat_mode
    valid_modes = ['chat', 'rag', 'wiki', 'write']
    if mode.lower() in valid_modes:
        chat_mode = mode.lower()
        output_text = f"Mode is now set to {chat_mode}"
    else:
        output_text = f'Modes are {", ".join(valid_modes)}'
    return send_back_message(user_id, output_text)

continue_text=""
def generate_response(message, user_id):
    global chat_mode, continue_text
    input_text=f"{message}"
    if chat_mode == 'chat':
        if message.startswith("/mode"):
            mode = message.replace("/mode", "").strip()
            return set_chat_mode(mode, user_id)

        elif message.startswith("/commands"):
            output = "/mode : <chat(active)|rag|wiki|write>\n""/continue : write mode only. passes previous message directly back into llm"
            return send_back_message(user_id, output)

        elif message.startswith("/continue"):
            output = "/continue only works in write mode"
            return send_back_message(user_id, output)

        elif message.startswith("/"):
            output_text = "Command not recognized. Available commands: /commands  /mode  /continue"
            return send_back_message(user_id, output)

        else:
            def generate_message():
                output = chat_chain.run(input=input_text)
                send_back_message(user_id, str(output))
                ENTITY_STORE.save_to_file(file_path="./memory_db/Chroma/entity_memory")
            threading.Thread(target=generate_message).start()
            return "..."

    if chat_mode == 'rag':
        if message.startswith("/mode"):
            mode = message.replace("/mode", "").strip()
            return set_chat_mode(mode, user_id)

        elif message.startswith("/commands"):
            output = "/mode : <chat|rag(active)|wiki|write>\n""/continue : write mode only. passes previous message directly back into llm"
            return send_back_message(user_id, output)

        elif message.startswith("/continue"):
            output = "/continue only works in write mode"
            return send_back_message(user_id, output)

        elif message.startswith("/"):
            output_text = "Command not recognized. Available commands: /commands  /mode  /continue"
            return send_back_message(user_id, output_text)

        else:
            def generate_message():
                output = rag_chain.run(input=input_text)
                send_back_message(user_id, str(output))
            threading.Thread(target=generate_message).start()
            return "..."

    if chat_mode == 'wiki':
        if message.startswith("/mode"):
            mode = message.replace("/mode", "").strip()
            return set_chat_mode(mode, user_id)

        elif message.startswith("/commands"):
            output = "/mode : <chat|rag|wiki(active)|write>\n""/continue : write mode only. passes previous message directly back into llm"
            return send_back_message(user_id, output)

        elif message.startswith("/continue"):
            output = "/continue only works in write mode"
            return send_back_message(user_id, output)

        elif message.startswith("/"):
            output_text = "Command not recognized. Available commands: /commands  /mode  /continue"
            return send_back_message(user_id, output_text)

        else:
            chat_history=[]
            def generate_message():
                output_text = wiki_chain({"question": input_text, "chat_history": chat_history})
                output=output_text['answer']
                send_back_message(user_id, str(output))
            threading.Thread(target=generate_message).start()
            return "..."

    if chat_mode == 'write':
        if message.startswith("/mode"):
            mode = message.replace("/mode", "").strip()
            return set_chat_mode(mode, user_id)

        elif message.startswith("/commands"):
            output = "/mode : <chat|rag|wiki|write(active)>\n""/continue : write mode only. passes previous message directly back into llm"
            return send_back_message(user_id, output)

        elif message.startswith("/continue"):
            def generate_message(): 
                global continue_text
                output = llm(continue_text)
                continue_text = output
                send_back_message(user_id, str(output))
            threading.Thread(target=generate_message).start()
            return "..."

        elif message.startswith("/"):
            output_text = "Command not recognized. Available commands: /commands  /mode  /continue"
            return send_back_message(user_id, output_text)

        else:
            def generate_message():
                global continue_text
                output = llm(input_text)
                continue_text = output
                send_back_message(user_id, str(output))
            threading.Thread(target=generate_message).start()
            return "..."

@app.route('/SynologyLLM', methods=['POST'])
def chatbot():
    token = SYNOCHAT_TOKEN
    webhook = OutgoingWebhook(request.form, token)
    if not webhook.authenticate(token):
        return webhook.createResponse('Outgoing Webhook authentication failed: Token mismatch.')
    message = webhook.text
    user_id = webhook.user_id
    task_queue.put((message, user_id))
    return "Task queued for processing"

def process_tasks():
    while True:
        processing_semaphore.acquire()
        try:
            message, user_id = task_queue.get()
            generate_response(message, user_id)
        finally:
            task_queue.task_done()

task_thread = threading.Thread(target=process_tasks, daemon=True)
task_thread.start()

if __name__ == '__main__':
    initialize_model()
    processing_thread = threading.Thread(target=process_tasks, daemon=True)
    processing_thread.start()
    app.run('0.0.0.0', port=FLASK_PORT, debug=False, threaded=True)