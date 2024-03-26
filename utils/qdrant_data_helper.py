    
from pydantic import BaseModel, Field
from typing import Any, Optional, List
from llama_index.core import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# pip install llama-index-vector-stores-qdrant llama-index-readers-file llama-index-embeddings-fastembed llama-index-llms-openai
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.ollama import Ollama


class Query(BaseModel):
    """
    主要用於接收使用者的查詢
    :param query: 使用者的查詢
    :param similarity_top_k: 查詢的相似度前幾名
    """
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=5)

class Response(BaseModel):
    """
    主要用於回傳給使用者的回應
    :param search_result: 搜尋結果
    :param source: 搜尋結果的來源
    """
    search_result: str 
    source: str


class DataIngestor:
    def __init__(self, embedder_name="sentence-transformers/all-mpnet-base-v2", collection_name="test_collection", q_client_url="http://localhost:6333/", q_api_key=None, data_path="./data"):
        """
        :param embedder_name: 嵌入模型的名稱
        :param collection_name: 資料集名稱
        :param q_client_url: Qdrant client 的 URL
        :param q_api_key: Qdrant client 的 API key
        :param data_path: 資料集的路徑(folder path)
        """
        self.embedder = HuggingFaceEmbedding(model_name=embedder_name)
        self.collection_name = collection_name
        if q_api_key is None:
            self.client = qdrant_client.QdrantClient(url=q_client_url)
        else:
            self.client = qdrant_client.QdrantClient(url=q_client_url, api_key=q_api_key)
        self.data_path = data_path

    def load_data(self):
        return SimpleDirectoryReader(self.data_path).load_data()

    def create_storage_context(self):
        qdrant_vector_store = QdrantVectorStore(client=self.client, collection_name=self.collection_name)
        return StorageContext.from_defaults(vector_store=qdrant_vector_store)

    def create_service_context(self):
        return ServiceContext.from_defaults(llm=None,embed_model=self.embedder, chunk_size=1024)

    def ingest(self):
        """
        用於將資料集載入 Qdrant 中
        """
        documents = self.load_data()
        storage_context = self.create_storage_context()
        service_context = self.create_service_context()

        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)
        return index
    
class RAG:
    def __init__(self,embedder_name="sentence-transformers/all-mpnet-base-v2", q_client_url="http://localhost:6333/", q_api_key="None", ollama_model="gemma:7b", ollama_base_url="http://localhost:11434"):
        """
        主要是用於建立 RAG 模型
        :param llm: ollama llm
        """

    
        if q_api_key is None:
            self.client = qdrant_client.QdrantClient(url=q_client_url)
        else:
            self.client = qdrant_client.QdrantClient(url=q_client_url, api_key=q_api_key)

        llm = Ollama(model=ollama_model)
        llm.base_url = ollama_base_url
        self.llm = llm  # ollama llm
        self.embedder_name = embedder_name
        self.embed_model = self.load_embedder()  # 加載嵌入模型

    def load_embedder(self):
        return HuggingFaceEmbedding(model_name=self.embedder_name)

    def qdrant_index(self, collection_name="dcard_collection", chunk_size=1024):
        """
        用於建立 Qdrant 的索引
        :param collection_name: 資料集名稱
        :param chunk_size: chunk size
        """
        qdrant_vector_store = QdrantVectorStore(client=self.client, collection_name=collection_name)
        service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model, chunk_size=chunk_size)

        index = VectorStoreIndex.from_vector_store(vector_store=qdrant_vector_store, service_context=service_context)
        return index

    def __get__response__(self, 
                          index: VectorStoreIndex, 
                          query: Query
                          , append_query: str = ""
                          ) -> Response:
        """
        主要是用於查詢
        :param __index: 索引
        :param __query: 查詢
        :param __append_query: 附加查詢 一些中文字 或 英文字
        """
        assert index is not None, "Index is required"
        assert query is not None, "Query is required"
        assert type(query) == Query, "Query must be of type Query"
        assert type(index) == VectorStoreIndex, "Index must be of type VectorStoreIndex"

        query_engine = index.as_query_engine(similarity_top_k=query.similarity_top_k, output=Response, response_mode="tree_summarize", verbose=True)
        response = query_engine.query(query.query + append_query)
        response_object = Response(
            search_result=str(response).strip(), 
            source=[response.metadata[s]["file_path"] for s in response.metadata.keys()][0]
        )
        return response_object
