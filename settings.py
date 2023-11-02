MODEL_FILENAME = "dolphin-llama2-7b.Q4_K_M.gguf" #Model used for testing
CONTEXT_LENGTH = 4096
GPU_LAYERS = None
LORA_BASE = None
LORA_PATH = None
MAX_TOKENS = 1000
TEMPURATURE = 0.92
TOP_P = 0.5
TOP_K = 100
STOP_WORDS=["Human:", "human:", "Question:", "question:"]
REPEAT_PENALTY = 1.3

CHAT_MODE = "chat" #chat, rag, wiki, write (default mode to boot into)

FLASK_PORT = 5015
SYNOCHAT_TOKEN = "Synology Token"
INCOMING_WEBHOOK_URL = "Bot Incoming Webhook Url"
