# Local-Qdrant-RAG

Local-Qdrant-RAG is a framework designed to leverage the powerful combination of Qdrant for vector search and RAG (Retrieval-Augmented Generation) for enhanced query understanding and response generation. It uses the Qdrant service for storing and retrieving vector embeddings and the RAG model to augment query responses with information retrieved from Qdrant.

## Getting Started

This guide will help you set up Local-Qdrant-RAG on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- Docker and Docker Compose (for running Qdrant locally)
- Ollama (for running LLM locally)
- Qdrant (Vector Database)

### Installation

1. **Clone the Repository**
    
    ```bash
    $git clone <https://github.com/your-repo/local-qdrant-rag.git>
    $cd local-qdrant-rag
    ```
    
2. **Install Dependencies**
    
    Install all necessary Python packages.

    ```bash
    $pip install -r requirements.txt
    ```

3. **Set Up Qdrant**

    Follow the instructions in the [Qdrant website](https://qdrant.tech/)

    ```bash
    docker pull qdrant/qdrant
    docker run -p 6333:6333 qdrant/qdrant
    ```

4. **Set Up Ollama**

    Follow the instructions in the [Ollama website](https://ollama.com/)

    * Download the Ollama CLI
    ```bash
    $curl -fsSL https://ollama.com/install.sh | sh
    ```
    * Download the Ollama model [here](https://ollama.com/library)
    * Run the Ollama model
    ```bash
    $ollama run <path-to-model>
    ```

### **Usage**

1. **Ingest Data into Qdrant**
    
    Before using the RAG model, you need to ingest your dataset into Qdrant. The **`DataIngestor`** class in **`qdrant_data_helper.py`** can help with this task.

    ```python
    from utils import qdrant_data_helper

    ingestor = qdrant_data_helper.DataIngestor(
        q_client_url="http://localhost:6333/", 
        q_api_key="test", # you can change this to your own qdrant api key if you have set it, otherwise, using None
        data_path="./data/", 
        collection_name="dcard_collection", 
        embedder_name="sentence-transformers/all-mpnet-base-v2"
        )

    index = ingestor.ingest()

    print("Index created successfully!")
    ```

    Folder structure for data_path:
    ```
    data/
    ├── doc1.txt
    ├── doc2.txt
    ├── doc3.txt
    └── ...
    ```

2. **Using the RAG Model**

    To use the RAG model for query responses, instantiate the RAG class and use the get_response method.
    
    ```python
    """
    Thie is inference code for the RAG and Qdrant with Ollama
    """

    from utils.qdrant_data_helper import RAG, Query

    def main():
        host = "localhost"
        rag = RAG(
            q_client_url=f"http://{host}:6333/", 
            q_api_key="test", # you can change this to your own qdrant api key if you have set it, otherwise, using None
            ollama_model="gemma:7b", 
            ollama_base_url=f"http://{host}:11434",
            )
        
        search_index = rag.qdrant_index(
                        collection_name="dcard_collection", 
                        chunk_size=1024
                        )

        query = Query(
                query="高科大是什麼時候合併的？",
                top_k=5
        )

        result = rag.get_response(
                        index= search_index,
                        query= query,
                        append_query="",
                        response_mode="tree_summarize"
                    )

        print("Result: ", result.search_result)
        print("Score: ", result.source)

    if __name__ == "__main__":
        main()
    ```

    Follow the llamaindex library to set response_mode. [llamaindex](https://docs.llamaindex.ai/en/stable/)


## References

- [Qdrant](https://qdrant.tech/)
- [OLLAMA](https://ollama.com/)
- [LLAMAINDEX](https://docs.llamaindex.ai/en/stable/)
- [Docker](https://www.docker.com/)
- [Hugging Face](https://huggingface.co/)

