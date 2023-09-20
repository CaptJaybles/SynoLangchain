from flask import Flask, request
import json
import os
import time
import requests
import threading
from synology import OutgoingWebhook
from settings import *
from CustomOutputParser import CustomOutputParser
from CustomPromptTemplate import CustomPromptTemplate
from VectorMemory import LoadCoreMemories, VECTORSTORE, VECTORSTORE2
from langchain.llms import LlamaCpp
from langchain.memory import VectorStoreRetrieverMemory, ConversationBufferWindowMemory, CombinedMemory, ReadOnlySharedMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper

app = Flask(__name__)

output_parser = CustomOutputParser()
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

#llama-cpp-python
llm = LlamaCpp(
    model_path=f"./model/{MODEL_FILENAME}",
    n_ctx=CONTEXT_LENGTH,
    n_gpu_layers=GPU_LAYERS,
    max_tokens=MAX_TOKENS,
    temperature=TEMPURATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    repeat_penalty=REPEAT_PENALTY,
    callback_manager=callback_manager,
    verbose=True
)

#Tools the agents can use
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to answer general questions about people, places, companies, historical events, or other subjects. Input should be a search query.",
        return_direct=False,
    ),
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
]

#vector memory using spacy and Chroma
LoadCoreMemories()
retriever = VECTORSTORE.as_retriever(search_type="mmr", search_kwargs={'k': 1, 'lambda_mult': 0.25})
retriever2 = VECTORSTORE2.as_retriever(search_type="mmr", search_kwargs={'k': 1, 'lambda_mult': 0.75})

#combined conversation and vector memory
knowledge_mem = VectorStoreRetrieverMemory(exclude_input_keys=["chat_history", "history"], input_key="input", memory_key="knowledge", retriever=retriever2)
readonlymemory = ReadOnlySharedMemory(memory=knowledge_mem)

long_memory = VectorStoreRetrieverMemory(exclude_input_keys=["chat_history", "knowledge"], input_key="input", retriever=retriever)
short_memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", input_key="input")
memory = CombinedMemory(memories=[short_memory, long_memory, readonlymemory])

#Custom Prompt for langchain
prompt = CustomPromptTemplate(
    template=PROMPT_TEMPLATE,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history", "chat_history", "knowledge"],
)

#Langchain chains and agents implemetation
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=STOP_WORDS,
    allowed_tools=tool_names,
    verbose=True
)

agent_chain = AgentExecutor.from_agent_and_tools(agent=agent,
    tools=tools, 
    callback_manager=callback_manager, 
    max_iterations=3,
    memory=memory, 
    early_stopping_method="generate", 
    verbose=False
)

#sending response back to synology
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
    return "success"

#message generating code
def initialize_model():
    warmup_input = f"What are the names of the planets?"
    agent_chain.run(input=warmup_input)

def generate_response(message, user_id):
    input_text = f"{message}"
    def generate_message():
        output_text = agent_chain.run(input=input_text)
        send_back_message(user_id, output_text)
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
    return generate_response(message, user_id,)

if __name__ == '__main__':
    initialize_model()
    app.run('0.0.0.0', port=FLASK_PORT, debug=False, threaded=True, processes=1)