import importlib.util
from typing import Any, Dict, List
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from settings import SPACY_MODEL
import spacy

nlp = spacy.load(SPACY_MODEL)

class SpacyEmbeddings(BaseModel, Embeddings):
    """Embeddings by SpaCy models.

    Attributes:
        nlp (Any): The Spacy model loaded into memory.

    Methods:
        embed_documents(texts: List[str]) -> List[List[float]]:
            Generates embeddings for a list of documents.
        embed_query(text: str) -> List[float]:
            Generates an embedding for a single piece of text.
    """

    nlp: Any  # The Spacy model loaded into memory

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid  # Forbid extra attributes during model initialization

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validates that the Spacy package and the SPACY_MODEL model are installed.

        Args:
            values (Dict): The values provided to the class constructor.

        Returns:
            The validated values.

        Raises:
            ValueError: If the Spacy package or the SPACY_MODEL
            model are not installed.
        """
        # Check if the Spacy package is installed
        if importlib.util.find_spec("spacy") is None:
            raise ValueError(
                "Spacy package not found. "
                "Please install it with `pip install spacy`."
            )
        try:
            # Try to load the SPACY_MODEL Spacy model
            import spacy

            values["nlp"] = spacy.load(SPACY_MODEL)
        except OSError:
            # If the model is not found, raise a ValueError
            raise ValueError(
                "Spacy model SPACY_MODEL not found. "
                "Please install it with"
                " `python -m spacy download SPACY_MODEL`."
            )
        return values  # Return the validated values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            texts (List[str]): The documents to generate embeddings for.

        Returns:
            A list of embeddings, one for each document.
        """
        return [self.nlp(text).vector.tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single piece of text.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            The embedding for the text.
        """
        return self.nlp(text).vector.tolist()

from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import SpacyTextSplitter

embeddings = SpacyEmbeddings()

VECTORSTORE = Chroma(persist_directory="./memory_db/Chroma", embedding_function=embeddings, collection_name='Memory')
VECTORSTORE2 = Chroma(embedding_function=embeddings, collection_name='Knowledge')

class LoadCoreMemories():
    mem_path = "./memory_db/core_memories"
    knowledge_path = "./memory_db/knowledge"

    mem_loader = DirectoryLoader(mem_path, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = PyPDFDirectoryLoader(knowledge_path)
    txt_loader = DirectoryLoader(knowledge_path, glob="**/*.txt")

    core_mem = mem_loader.load()
    txt_knowledge = txt_loader.load()
    pdf_knowledge = pdf_loader.load()
    text_splitter = SpacyTextSplitter(chunk_size=1000)
    knowledge_txt = text_splitter.split_documents(txt_knowledge)

    VECTORSTORE.add_documents(core_mem)
    VECTORSTORE2.add_documents(knowledge_txt)
    VECTORSTORE2.add_documents(pdf_knowledge)