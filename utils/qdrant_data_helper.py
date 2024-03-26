from typing import List, Optional
import qdrant_client
from .format import Query, Response
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext, ServiceContext, VectorStoreIndex, get_response_synthesizer
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate

# Local settings
from llama_index.core.node_parser import SentenceSplitter

class DataIngestor:
    """Utility for ingesting a dataset into Qdrant.

    Attributes:
        embedder: An embedding model for generating vector representations of documents.
        collection_name: The name of the collection in Qdrant.
        client: The Qdrant client for communicating with the Qdrant service.
        data_path: The local storage path of the dataset.
    """
    def __init__(self, embedder_name: str = "sentence-transformers/all-mpnet-base-v2", collection_name: str = "test_collection", q_client_url: str = "http://localhost:6333/", q_api_key: Optional[str] = None, data_path: str = "./data", chunk_size: int = 1024):
        """
        Initializes an instance of DataIngestor.

        Args:
            embedder_name: The name of the embedding model.
            collection_name: The name of the dataset collection.
            q_client_url: The URL for the Qdrant client.
            q_api_key: The API key for the Qdrant client.
            data_path: The path (folder path) of the dataset.
        """
        self.embedder = HuggingFaceEmbedding(model_name=embedder_name,max_length=512)
        self.collection_name = collection_name
        self.client = qdrant_client.QdrantClient(url=q_client_url, api_key=q_api_key) if q_api_key else qdrant_client.QdrantClient(url=q_client_url)
        self.data_path = data_path
        self.chunk_size = chunk_size

    def load_data(self) -> List[dict]:
        """Loads data from the specified data path.

        Returns:
            A list of dictionaries containing the data.
        """
        return SimpleDirectoryReader(self.data_path).load_data()

    def create_storage_context(self) -> StorageContext:
        """Creates and configures a storage context.

        Returns:
            The configured storage context.
        """
        qdrant_vector_store = QdrantVectorStore(client=self.client, collection_name=self.collection_name)
        return StorageContext.from_defaults(vector_store=qdrant_vector_store)

    def create_service_context(self) -> ServiceContext:
        """Creates and configures a service context.

        Returns:
            The configured service context.
        """
        return ServiceContext.from_defaults(llm=None, embed_model=self.embedder, chunk_size=self.chunk_size)

    def ingest(self):
        
        """Ingests the dataset into Qdrant."""
        documents = self.load_data()
        storage_context = self.create_storage_context()
        service_context = self.create_service_context()

        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            service_context=service_context,
            transformations=[SentenceSplitter(chunk_size=self.chunk_size, separator=",")]
        )
        return index

   
class RAG:
    """A class for constructing a Retrieval-Augmented Generation (RAG) model.

    This class is responsible for setting up the components needed for a RAG model, including the embedding model, Qdrant client, and the Ollama language model.

    Attributes:
        embedder_name: The name of the embedding model.
        q_client_url: The URL for the Qdrant client.
        q_api_key: The API key for the Qdrant client (optional).
        ollama_model: The name of the Ollama model.
        ollama_base_url: The base URL for the Ollama model.
        client: The Qdrant client.
        llm: The Ollama language model.
        embed_model: The loaded embedding model.
    """
    def __init__(self, embedder_name: str = "sentence-transformers/all-mpnet-base-v2", q_client_url: str = "http://localhost:6333/", q_api_key: Optional[str] = None, ollama_model: str = "gemma:7b", ollama_base_url: str = "http://localhost:11434", SYSTEM_PROMPT: str = None):
        """
        Initializes the RAG model with the necessary components.

        Args:
            embedder_name: The name of the embedding model to use.
            q_client_url: The URL to connect to the Qdrant service.
            q_api_key: An optional API key for authenticated access to Qdrant.
            ollama_model: The name of the Ollama model to use for generation.
            ollama_base_url: The base URL for accessing the Ollama service.
            SYSTEM_PROMPT: The system prompt to use for the RAG model (optional).
        """
        SYSTEM_PROMPT = """
        You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
        - Generate human readable output, avoid creating output with gibberish text.
        - Generate only the requested output, don't include any other language before or after the requested output.
        - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
        - Generate professional language typically used in business documents in North America.
        - Never generate offensive or foul language.
        """ if SYSTEM_PROMPT is None else SYSTEM_PROMPT

        self.query_wrapper_prompt = PromptTemplate(
            "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
        )

        self.client = qdrant_client.QdrantClient(url=q_client_url, api_key=q_api_key) if q_api_key else qdrant_client.QdrantClient(url=q_client_url)
        self.llm = Ollama(
            model=ollama_model, 
            base_url=ollama_base_url,
        )
        self.embedder_name = embedder_name
        self.embed_model = self.load_embedder()
        
    def load_embedder(self) -> HuggingFaceEmbedding:
        """Loads the embedding model.

        Returns:
            The loaded HuggingFaceEmbedding model.
        """
        return HuggingFaceEmbedding(model_name=self.embedder_name)

    def qdrant_index(self, collection_name: str = "dcard_collection", chunk_size: int = 1024) -> VectorStoreIndex:
        """Creates an index in Qdrant for the specified collection.

        Args:
            collection_name: The name of the collection to index.
            chunk_size: The size of chunks for processing documents.

        Returns:
            The VectorStoreIndex object for the specified collection.
        """
        qdrant_vector_store = QdrantVectorStore(client=self.client, collection_name=collection_name)
        service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model, chunk_size=chunk_size)
        return VectorStoreIndex.from_vector_store(vector_store=qdrant_vector_store, service_context=service_context)

    def get_response(self, index: VectorStoreIndex, query: 'Query', append_query: str = "", response_mode:str ="tree_summarize") -> 'Response':
        """Executes a query using the RAG model.

        Args:
            index: The VectorStoreIndex to use for querying.
            query: The Query object containing the user's query.
            append_query: Additional text to append to the query (optional).

        Returns:
            A Response object containing the search result and source.

        Raises:
            AssertionError: If any of the inputs are not of the expected type.
        """
        assert index is not None and isinstance(index, VectorStoreIndex), "Index must be provided and be of type VectorStoreIndex."
        assert query is not None and isinstance(query, Query), "Query must be provided and be of type Query."

        # configure response synthesizer

        query_engine = index.as_query_engine(similarity_top_k=query.similarity_top_k, output='Response', response_mode=response_mode, prompt_template=self.query_wrapper_prompt)
        response = query_engine.query(query.query + append_query)
        response_object = Response(
            search_result=str(response).strip(), 
            source=[response.metadata[s]["file_path"] for s in response.metadata.keys()][0]
        )
        return response_object