import json
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

from src.services import get_embedder
from src.config import get_config

def create_collection():
    collection_name = "scenario_components"
    
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists - will append new chunks.")
        collection = Collection(name=collection_name)
        try:
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
        except Exception:
            pass
        return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="scenario_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="component_type", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    return collection

def load_json_file(filepath: str) -> List[Dict]:
    """Reads a single JSON file containing a list of scenario components."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Just in case the JSON is a single dictionary instead of a list, wrap it
    if isinstance(data, dict):
        data = [data]
        
    return data

def insert_scenarios(collection: Collection, data_items: List[Dict], embedding_model: Any, batch_size: int = 32):
    inserted_count = 0
    skipped_count = 0
    
    batch_scenario_ids = []
    batch_component_types = []
    batch_descriptions = []
    batch_codes = []
    batch_texts_for_embedding = []
    
    print(f"Starting insertion for {len(data_items)} items...")
    
    for idx, item in enumerate(data_items):
        # Extract fields directly from the flat JSON object
        scenario_id = item.get("scenario_id", "")
        component_type = item.get("component_type", "")
        description = item.get("description", "")
        code = item.get("code", "")
        
        # Clean up text
        description = description.strip() if description else ""
        code = code.strip() if code else ""
        
        # Skip if either is empty (based on your previous logic)
        if not description or not code:
            skipped_count += 1
            continue
            
        # Add to current batch
        batch_scenario_ids.append(scenario_id)
        batch_component_types.append(component_type)
        batch_descriptions.append(description)
        batch_codes.append(code)
        batch_texts_for_embedding.append({"text": description}) # We only embed the description
        
        # If batch is full, insert it
        if len(batch_texts_for_embedding) >= batch_size:
            try:
                embeddings = embedding_model.encode(batch_texts_for_embedding)
                
                data_to_insert = [
                    batch_scenario_ids,
                    batch_component_types,
                    batch_descriptions,
                    batch_codes,
                    embeddings
                ]
                
                collection.insert(data_to_insert)
                inserted_count += len(batch_texts_for_embedding)
                print(f"Processed {idx + 1}/{len(data_items)} items...")
                
            except Exception as e:
                print(f"Error inserting batch ending at index {idx}: {e}")
                skipped_count += len(batch_texts_for_embedding)
            
            # Reset batches
            batch_scenario_ids = []
            batch_component_types = []
            batch_descriptions = []
            batch_codes = []
            batch_texts_for_embedding = []

    # Insert any leftover items in the final batch
    if batch_texts_for_embedding:
        try:
            embeddings = embedding_model.encode(batch_texts_for_embedding)
            
            data_to_insert = [
                batch_scenario_ids,
                batch_component_types,
                batch_descriptions,
                batch_codes,
                embeddings
            ]
            
            collection.insert(data_to_insert)
            inserted_count += len(batch_texts_for_embedding)
        except Exception as e:
            print(f"Error inserting final batch: {e}")
            skipped_count += len(batch_texts_for_embedding)
            
    print(f"Finished! Inserted: {inserted_count}, Skipped: {skipped_count}")

def main():
    config = get_config()
    connections.connect(uri=f"http://{config.vector_db.host}:{config.vector_db.port}")
    
    device = "cuda"
    embedding_model = get_embedder(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    print(f"Using device: {device}")
    
    collection = create_collection()
    
    json_filepath = "data/raw_snippets/recovered_scenario_components_with_subject.json"
    scenarios = load_json_file(json_filepath)
    
    print(f"Inserting {len(scenarios)} scenarios into Milvus...")
    insert_scenarios(collection, scenarios, embedding_model, batch_size=32)
    
    # FIXED: Flush ONCE right before loading the collection
    print("Flushing data to disk...")
    collection.flush()
    
    print("Loading collection into memory...")
    collection.load()
    
    # FIXED: Dynamically use the actual collection name so it's always accurate
    print(f"Inserted {collection.num_entities} components into collection '{collection.name}'")
    
    connections.disconnect("default")

if __name__ == "__main__":
    main()