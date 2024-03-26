"""
Thie is inference code for the RAG and Qdrant with Ollama
"""

from utils.qdrant_data_helper import RAG, Query

def main():
    QA_Prompt_summarization_Prompt = """
        You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
        - Generate human readable output, avoid creating output with gibberish text.
        - Generate only the requested output, don't include any other language before or after the requested output.
        - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
        - Generate professional language typically used in business documents in North America.
        - Never generate offensive or foul language.
        """
    host = "localhost"
    rag = RAG(
        q_client_url=f"http://{host}:6333/", 
        q_api_key="test", # you can change this to your own qdrant api key if you have set it, otherwise, using None
        ollama_model="gemma:7b", 
        ollama_base_url=f"http://{host}:11434",
        SYSTEM_PROMPT=QA_Prompt_summarization_Prompt
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
    
