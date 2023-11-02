from datetime import datetime
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

metadata = {'timestamp': datetime.now().isoformat()}
embedding_function = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}, cache_folder='./model')
VECTORSTORE = Chroma(persist_directory="./memory_db/Chroma/memory", embedding_function=embedding_function, collection_name='Memory', collection_metadata=metadata)
VECTORSTORE2 = Chroma(embedding_function=embedding_function, collection_name='Knowledge', collection_metadata=metadata)

class LoadCoreMemories():
    mem_path = "./memory_db/core_memories"
    knowledge_path = "./memory_db/knowledge"
    mem_loader = DirectoryLoader(mem_path, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(knowledge_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(knowledge_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    core_mem = mem_loader.load()
    txt_knowledge = txt_loader.load()
    pdf_knowledge = pdf_loader.load()
    VECTORSTORE.add_documents(core_mem)
    try:
        knowledge_pdf = text_splitter.split_documents(pdf_knowledge)
        VECTORSTORE2.add_documents(knowledge_pdf)
    except:
        pass
    try:
        knowledge_txt = text_splitter.split_documents(txt_knowledge)
        VECTORSTORE2.add_documents(knowledge_txt)
    except:
        pass

from typing import Any, Dict, Iterable, List, Optional
import pickle
import os
from langchain.memory.entity import BaseEntityStore

class InMemoryEntityStore(BaseEntityStore):
    """In-memory Entity store."""

    store: Dict[str, Optional[str]] = {}

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.store.get(key, default)

    def set(self, key: str, value: Optional[str]) -> None:
        self.store[key] = value

    def delete(self, key: str) -> None:
        del self.store[key]

    def exists(self, key: str) -> bool:
        return key in self.store

    FILE_NAME = "entity_store.pkl"

    def save_to_file(self, file_path: str) -> None:
        """Save the entity store to a file."""
        full_file_path = os.path.join(file_path, self.FILE_NAME)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

        with open(full_file_path, 'wb') as file:
            pickle.dump(self.store, file)

    def load_from_file(self, file_path: str) -> None:
        """Load the entity store from a file."""
        full_file_path = os.path.join(file_path, self.FILE_NAME)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

        if os.path.exists(full_file_path):
            with open(full_file_path, 'rb') as file:
                data = pickle.load(file)
                self.store.update(data)

    def clear(self) -> None:
        return self.store.clear()

ENTITY_STORE=InMemoryEntityStore()
ENTITY_STORE.load_from_file(file_path="./memory_db/Chroma/entity_memory")