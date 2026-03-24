from pymilvus import MilvusClient

def reverse_engineer_milvus(uri: str):
    print(f"Connecting to Milvus at: {uri}...")
    client = MilvusClient(uri=uri)
    
    # 1. Discover all collections
    collections = client.list_collections()
    
    if not collections:
        print("Database connected successfully, but it is completely empty (0 collections).")
        return

    print(f"Found {len(collections)} collections: {collections}\n")

    for col_name in collections:
        print(f"{'='*50}")
        print(f"🔍 COLLECTION: {col_name}")
        print(f"{'='*50}")

        # 2. Extract the exact Schema and Fields
        details = client.describe_collection(collection_name=col_name)
        
        print("FIELDS:")
        for field in details.get('fields', []):
            is_pk = " (PRIMARY KEY)" if field.get('is_primary') else ""
            
            # If it's a vector, grab the dimensions. If text, grab max length.
            params = field.get('params', {})
            dim = f" | Dim: {params.get('dim')}" if 'dim' in params else ""
            max_len = f" | Max Length: {params.get('max_length')}" if 'max_length' in params else ""
            
            print(f"  - {field['name']}: {field['type']}{is_pk}{dim}{max_len}")

        # 3. Extract Index Information
        print("\nINDEXES:")
        indexes = client.list_indexes(collection_name=col_name)
        for idx in indexes:
            index_details = client.describe_index(collection_name=col_name, index_name=idx)
            print(f"  - Field: {index_details.get('field_name')} | Type: {index_details.get('index_type')} | Metric: {index_details.get('metric_type')}")

        # 4. Get the total number of vectors/rows
        stats = client.get_collection_stats(collection_name=col_name)
        row_count = stats.get('row_count', 'Unknown')
        print(f"\nTOTAL ROWS: {row_count}\n")

def fix_milvus_index(uri: str, collection_name: str):
    print(f"Connecting to {uri}...")
    client = MilvusClient(uri=uri)
    
    # 1. Force the database to drop the corrupted index
    print(f"Dropping the old Hugging Face index...")
    try:
        client.drop_index(collection_name=collection_name, index_name="embedding")
        print("Old index dropped successfully.")
    except Exception as e:
        print(f"Note: Could not drop index (it might not exist). Error: {e}")
    
    # 2. Rebuild a fresh index using YOUR computer's hardware
    print(f"Building a fresh IVF_FLAT index for {collection_name}...")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding", 
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128}
    )
    
    client.create_index(collection_name=collection_name, index_params=index_params)
    print("New index built successfully!")
    
    # 3. Try to load it into RAM again
    print("Attempting to load the collection into RAM...")
    client.load_collection(collection_name=collection_name)
    print("✅ Success! The deadlock is broken. You can now query the data.")

def read_elements_from_collection(uri: str, collection_name: str):
    print(f"Connecting to Milvus at: {uri}...")
    client = MilvusClient(uri=uri)

    print(f"\n{'='*50}")
    print(f"🚀 FETCHING FIRST 3 ELEMENTS FROM: {collection_name}")
    print(f"{'='*50}")
    
    try:
        client.load_collection(collection_name=collection_name)
        # The query hack: id >= 0 guarantees we get the first items without doing a vector search
        results = client.query(
            collection_name=collection_name,
            filter="id >= 0", 
            # We pull everything EXCEPT the embedding vector to keep the terminal clean
            output_fields=["id", "scenario_id", "component_type", "description", "code"],
            limit=3
        )
        
        for i, record in enumerate(results):
            print(f"\n--- Element {i+1} ---")
            print(f"Primary Key ID: {record.get('id')}")
            print(f"Scenario ID: {record.get('scenario_id')}")
            print(f"Component Type: {record.get('component_type')}")
            print(f"Description: {record.get('description')}")
            
            # Truncate the Scenic code to 150 characters so it doesn't break your screen
            code_str = str(record.get('code', ''))
            code_preview = (code_str[:150] + '...') if len(code_str) > 150 else code_str
            print(f"Code Preview: \n{code_preview.strip()}")
            
    except Exception as e:
        print(f"Error fetching data: {e}")

# Run it!
# Use "./milvus.db" for a local Lite file, or "http://localhost:19530" for Docker
if __name__ == "__main__":
    # reverse_engineer_milvus(uri="http://127.0.0.1:19530")
    # fix_milvus_index("http://127.0.0.1:19530", "scenario_components")
    read_elements_from_collection(uri="http://127.0.0.1:19530", collection_name="scenario_components")