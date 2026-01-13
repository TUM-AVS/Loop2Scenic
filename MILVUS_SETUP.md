# Milvus Setup Guide

This guide explains how to set up Milvus as your vector database for the ADS-MRAG pipeline.

## Why Milvus?

Milvus offers several advantages:

- ✅ **High Performance**: Optimized for billion-scale vector search
- ✅ **Advanced Filtering**: Powerful metadata and tag filtering with array support
- ✅ **Scalability**: Distributed architecture for production deployments
- ✅ **Multiple Index Types**: IVF_FLAT, IVF_SQ8, HNSW, and more
- ✅ **Production-Ready**: Battle-tested in large-scale applications

## Installation Options

### Option 1: Docker (Recommended for Development)

**1. Start Milvus using Docker Compose:**

```bash
# Start Milvus standalone
docker-compose up -d

# Check if Milvus is running
docker-compose ps
```

This will start:

- Milvus on port 19530
- MinIO (object storage) on ports 9000-9001
- etcd (metadata store) on port 2379

**2. Verify Milvus is running:**

```bash
# Check health
curl http://localhost:9091/healthz
```

**3. Configure the pipeline:**

The default `config/config.yaml` is already configured for Milvus:

```yaml
vector_db:
  provider: milvus
  host: localhost
  port: 19530
  collection_name: documents
  distance_metric: cosine
```

### Option 2: Milvus Lite (Lightweight, No Docker)

**1. Install Milvus Lite:**

```bash
pip install milvus
```

**2. Update configuration:**

Milvus Lite uses a local file instead of a server. Update `config/config.yaml`:

```yaml
vector_db:
  provider: milvus
  collection_name: documents
  distance_metric: cosine
```

Then in your code:

```python
from src.pipeline import RAGPipeline
from src.config import Config

config = Config.from_yaml()
config.vector_db.host = "localhost"  # Milvus Lite default

pipeline = RAGPipeline(config=config)
```

### Option 3: Milvus Cluster (Production)

For production deployments, see the [Milvus documentation](https://milvus.io/docs/install_cluster-milvusoperator.md).

## Configuration Options

### Basic Configuration

```yaml
vector_db:
  provider: milvus
  collection_name: documents
  host: localhost
  port: 19530
  distance_metric: cosine # cosine, l2, ip
```

### Advanced Configuration

```yaml
vector_db:
  provider: milvus
  collection_name: documents
  host: localhost
  port: 19530
  distance_metric: cosine

  # Index settings
  index_type: IVF_FLAT # IVF_FLAT, IVF_SQ8, HNSW
  nlist: 1024 # Number of cluster units
  nprobe: 10 # Number of clusters to search
```

### Index Types

| Index Type   | Description                        | Best For                      |
| ------------ | ---------------------------------- | ----------------------------- |
| **IVF_FLAT** | Inverted File with exact search    | Balanced performance/accuracy |
| **IVF_SQ8**  | IVF with scalar quantization       | Memory-efficient              |
| **HNSW**     | Hierarchical Navigable Small World | High-performance search       |

## Usage Examples

### Basic Usage

```python
from src.pipeline import RAGPipeline

# Initialize with Milvus
pipeline = RAGPipeline()

# Ingest documents with tags
pipeline.ingest_documents(
    source="data/raw/",
    tags=["finance", "2024"],
    is_directory=True
)

# Query with tag filtering
response = pipeline.query(
    "What were the Q1 results?",
    filter_tags=["finance"]
)

print(response)
```

### Advanced Filtering

```python
# Tag filtering (documents with python OR ml tags)
results = pipeline.retriever.retrieve(
    query="machine learning frameworks",
    filter_tags=["python", "ml"],
    return_scores=True
)

# Metadata filtering
results = pipeline.vectorstore.similarity_search(
    query="recent updates",
    k=5,
    metadata_filter={"year": 2024}
)

# Combined filtering (tags AND metadata)
results = pipeline.vectorstore.similarity_search(
    query="technical documentation",
    k=5,
    filter_tags=["python"],
    metadata_filter={"category": "tutorial"}
)
```

### Milvus Filter Expressions

Milvus uses string expressions for filtering:

```python
# Array contains
expr = 'array_contains(tags, "python")'

# Multiple tags (OR)
expr = 'array_contains(tags, "python") || array_contains(tags, "ml")'

# Metadata filtering
expr = 'year == 2024 && category == "tutorial"'

# Combined
expr = '(array_contains(tags, "python") || array_contains(tags, "ml")) && year == 2024'

# Use in search
results = pipeline.vectorstore.similarity_search(
    query="frameworks",
    k=5,
    metadata_filter=expr  # Pass raw expression
)
```

## Switching Between ChromaDB and Milvus

The pipeline supports both ChromaDB and Milvus. Switch by changing the `provider` in config:

### Use Milvus

```yaml
vector_db:
  provider: milvus
  host: localhost
  port: 19530
  collection_name: documents
```

### Use ChromaDB

```yaml
vector_db:
  provider: chromadb
  persist_directory: ./data/vector_db
  collection_name: documents
```

## Docker Commands

### Start Milvus

```bash
docker-compose up -d
```

### Stop Milvus

```bash
docker-compose down
```

### View Logs

```bash
docker-compose logs -f milvus
```

### Reset Milvus (Delete All Data)

```bash
docker-compose down -v
docker-compose up -d
```

### Access Milvus Admin UI

Milvus doesn't have a built-in UI, but you can use **Attu** (Milvus GUI):

```bash
docker run -p 8000:3000 -e MILVUS_URL=host.docker.internal:19530 zilliz/attu:latest
```

Then open http://localhost:8000

## Performance Tuning

### For Large Datasets (>1M vectors)

```yaml
vector_db:
  index_type: HNSW
  distance_metric: cosine
```

Update search params:

```python
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 64}  # Higher = more accurate but slower
}
```

### For Memory Efficiency

```yaml
vector_db:
  index_type: IVF_SQ8 # Scalar quantization
  nlist: 2048
```

### For Speed

```yaml
vector_db:
  index_type: HNSW
  distance_metric: ip # Inner product (fastest)
```

## Troubleshooting

### "Connection refused" Error

**Problem**: Cannot connect to Milvus

**Solutions**:

1. Check if Milvus is running: `docker-compose ps`
2. Verify port 19530 is accessible: `netstat -an | grep 19530`
3. Check Milvus logs: `docker-compose logs milvus`

### "Collection not found" Error

**Problem**: Collection doesn't exist

**Solution**: The collection is created automatically on first document insertion. Just add documents:

```python
pipeline.ingest_documents("data/raw/sample.txt", tags=["test"])
```

### Memory Issues

**Problem**: Milvus using too much memory

**Solutions**:

1. Use IVF_SQ8 instead of IVF_FLAT
2. Reduce batch size in config
3. Increase Docker memory limit

### Slow Search

**Problem**: Searches are slow

**Solutions**:

1. Increase `nprobe` for better results (trades speed)
2. Use HNSW index for faster search
3. Reduce `top_k` parameter

## Migration from ChromaDB

To migrate existing ChromaDB data to Milvus:

```python
from src.vectorstore import ChromaVectorStore, MilvusVectorStore
from src.embedding import Embedder

# Load from ChromaDB
embedder = Embedder()
chroma = ChromaVectorStore(embedder, persist_directory="./data/vector_db")

# Get all documents
# Note: You'll need to implement a method to export all documents
# Or re-ingest from original sources

# Add to Milvus
milvus = MilvusVectorStore(embedder)
# ... add documents to milvus
```

## Monitoring

### Check Collection Stats

```python
stats = pipeline.get_stats()
print(f"Documents: {stats['document_count']}")
print(f"Index type: {stats['index_type']}")
```

### Using Python Client

```python
from pymilvus import connections, Collection

# Connect
connections.connect(host="localhost", port="19530")

# Get collection
collection = Collection("documents")
collection.load()

# Stats
print(f"Entities: {collection.num_entities}")
print(f"Index: {collection.index().params}")
```

## Resources

- [Milvus Documentation](https://milvus.io/docs)
- [Milvus GitHub](https://github.com/milvus-io/milvus)
- [Attu (Milvus GUI)](https://github.com/zilliztech/attu)
- [Milvus Python SDK](https://milvus.io/docs/install-pymilvus.md)

## Next Steps

1. Start Milvus: `docker-compose up -d`
2. Test the pipeline: `python examples/basic_usage.py`
3. Ingest your documents: `python scripts/ingest_documents.py --source your_docs/ --directory`
4. Query: `python scripts/query_pipeline.py --query "your question"`

Happy searching with Milvus! 🚀
