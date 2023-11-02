#chat prompt
CHAT_TEMPLATE = """Relevant pieces of previous conversation:
{chat_history}

Context:
{entities}

{history}
Human: {input}
AI:"""



#pdf prompt
RAG_TEMPLATE = """Use the following pieces of information to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Relevant piece of information:
{knowledge}

Question: {input}
Answer:"""