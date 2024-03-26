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