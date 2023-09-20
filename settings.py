MODEL_FILENAME = "dolphin-llama2-7b.Q4_K_S.gguf" #Model used for testing
SPACY_MODEL = "en_core_web_md"
CONTEXT_LENGTH = 4000
GPU_LAYERS = None
LORA_BASE = None
LORA_PATH = None

FLASK_PORT = 5025
SYNOCHAT_TOKEN = 'Put_your_token_here'
INCOMING_WEBHOOK_URL = "Copy_from_synologychat_incoming_URL"

STOP_WORDS=["User:", "Human:", "Question:", "input:"]
MAX_TOKENS = 1000
TEMPURATURE = 0.92
TOP_P = 0.5
TOP_K = 100
REPEAT_PENALTY = 1.2

PROMPT_TEMPLATE = """Answer the question from the Human the best you can. You have access to the following tools:

{tools}

To use a tool, please use the following format:

Human: the input question you must answer
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 2 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

If you do not need to use a tool, you MUST use the format:

Human: the input you must respond to
Thought: Do I need to use a tool? No
AI: [your response here]

Begin!

Relevant piece of information:
{knowledge}

Relevant pieces of previous conversation:
{history}

Current conversation:
{chat_history}
Human: {input}
{agent_scratchpad}"""