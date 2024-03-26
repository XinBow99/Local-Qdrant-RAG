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
                     chunk_size=100
                     )

    query = Query(
            query="高科大是什麼時候合併的？",
            top_k=5
    )

    result = rag.get_response(
                    index= search_index,
                    query= query,
                    append_query="",
                    response_mode="refine"
                )

    print("Result: ", result.search_result)
    print("Score: ", result.source)

if __name__ == "__main__":
    main()

