import re
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)




class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            print(node.text)
        return nodes
    

def get_transform_pipeline(chunk_size: int=512, HFE_model_name: str="sentence-transformers/all-mpnet-base-v2"):
    """
    Returns a pipeline for transforming text data.

    Attributes:
        chunk_size: The size of the text chunks.
        HFE_model_name: The name of the Hugging Face embedding model.
    """
    transformations = [
        #TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=128),
        #TitleExtractor(),
        TextCleaner(),
        #HuggingFaceEmbedding(model_name=HFE_model_name,max_length=512),
        #QuestionsAnsweredExtractor(questions=3),
    ]
    return transformations

