# ADS-MRAG: Advanced Document Search - Metadata RAG Pipeline

A modular, production-ready RAG (Retrieval-Augmented Generation) pipeline with advanced tag-based filtering capabilities using ChromaDB as the vector database.

## 🌟 Features

- **Modular Architecture**: Clear separation of concerns with dedicated modules for each component
- **Tag-Based Filtering**: Filter documents by tags/metadata for precise retrieval
- **Multiple Document Formats**: Support for PDF, DOCX, TXT, Markdown, and HTML
- **Flexible Models**: Easy switching between HuggingFace/OpenAI embeddings and OpenAI/Anthropic LLMs
- **Milvus/ChromaDB Vector Store**: High-performance vector storage with metadata filtering
- **Configurable Pipeline**: YAML-based configuration for easy customization
- **Production-Ready**: Comprehensive logging, error handling, and testing

## 📁 Project Structure

```
ads-mrag/
├── config/                   # Configuration files
│   └── config.yaml          # Main configuration
├── data/                    # Data storage
│   ├── raw/                # Raw documents
│   ├── processed/          # Processed documents
│   └── vector_db/          # ChromaDB persistence
├── src/                     # Source code
│   ├── ingestion/          # Document loading and chunking
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   └── processor.py
│   ├── embedding/          # Embedding generation
│   │   └── embedder.py
│   ├── vectorstore/        # Vector database operations
│   │   └── chroma_store.py
│   ├── retrieval/          # Document retrieval
│   │   └── retriever.py
│   ├── generation/         # LLM response generation
│   │   ├── generator.py
│   │   └── prompts.py
│   ├── utils/              # Utility functions
│   │   ├── logger.py
│   │   └── helpers.py
│   ├── config.py           # Configuration management
│   └── pipeline.py         # Main pipeline orchestration
├── scripts/                 # Utility scripts
│   ├── ingest_documents.py
│   ├── query_pipeline.py
│   └── reset_vectorstore.py
├── examples/                # Example usage
│   ├── basic_usage.py
│   └── file_ingestion.py
├── notebooks/               # Jupyter notebooks
│   ├── 01_getting_started.ipynb
│   └── 02_advanced_filtering.ipynb
├── tests/                   # Unit tests
│   ├── test_config.py
│   └── test_chunker.py
├── logs/                    # Log files
├── .env.example            # Environment variables template
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda
- Docker (for Milvus - recommended) OR Milvus Lite

### Installation

1. **Clone the repository**

```bash
cd ads-mrag
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Start Milvus (vector database)**

```bash
# Start Milvus using Docker Compose
docker-compose up -d

# Verify it's running
docker-compose ps
```

> **Note**: For detailed Milvus setup instructions, see [MILVUS_SETUP.md](MILVUS_SETUP.md)

5. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required API keys:

- `OPENAI_API_KEY`: For OpenAI LLM (or use alternative providers)

### Quick Start

#### Python API

```python
from src.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ingest documents with tags
doc_ids = pipeline.ingest_documents(
    source="data/raw/my_documents",
    tags=["research", "2024"],
    is_directory=True
)

# Query with tag filtering
response = pipeline.query(
    query="What are the main findings?",
    filter_tags=["research"],
    return_sources=True
)

print(response["response"])
```

#### Command Line

**Ingest documents:**

```bash
python scripts/ingest_documents.py \
    --source data/raw/documents \
    --tags research,2024 \
    --directory
```

**Query the pipeline:**

```bash
python scripts/query_pipeline.py \
    --query "What are the main findings?" \
    --tags research \
    --show-sources
```

## 📚 Usage Examples

### 1. Basic Document Ingestion

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()

# Ingest a single file
pipeline.ingest_documents(
    source="document.pdf",
    tags=["finance", "report"],
    metadata={"year": 2024, "department": "sales"}
)

# Ingest a directory
pipeline.ingest_documents(
    source="documents/",
    tags=["technical", "documentation"],
    is_directory=True,
    recursive=True
)
```

### 2. Tag-Based Filtering

```python
# Query with tag filtering
response = pipeline.query(
    query="What are the sales figures?",
    filter_tags=["finance", "report"],
    top_k=5
)

# Retrieve documents by tags
results = pipeline.retriever.retrieve_by_tags(
    query="quarterly results",
    required_tags=["finance"],
    return_scores=True
)
```

### 3. Custom Configuration

```python
from src.config import Config

# Load custom config
config = Config.from_yaml("custom_config.yaml")
pipeline = RAGPipeline(config=config)
```

### 4. Processing Text Directly

```python
# Process texts without files
texts = [
    "Python is a programming language.",
    "Machine learning uses algorithms."
]
tags = [
    ["programming", "python"],
    ["ai", "ml"]
]

chunks = pipeline.processor.process_texts(
    texts=texts,
    tags=tags
)
pipeline.vectorstore.add_documents(chunks)
```

## ⚙️ Configuration

The pipeline is configured via `config/config.yaml`:

```yaml
vector_db:
  provider: milvus # milvus or chromadb
  host: localhost
  port: 19530
  collection_name: documents
  distance_metric: cosine

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
  device: cpu

chunking:
  chunk_size: 1000
  chunk_overlap: 200

retrieval:
  top_k: 5
  similarity_threshold: 0.7

llm:
  provider: openai
  model: gpt-3.5-turbo
  temperature: 0.7
```

## 🎯 Key Features Explained

### Tag-Based Filtering

Milvus supports powerful metadata filtering with array operations. You can:

- **Filter by single tag**: `filter_tags=["finance"]`
- **Filter by multiple tags**: `filter_tags=["finance", "2024"]` (OR operation)
- **Combine with metadata**: Filter by tags AND other metadata fields

Example:

```python
results = pipeline.vectorstore.similarity_search(
    query="quarterly results",
    filter_tags=["finance"],
    metadata_filter={"year": 2024},
    k=5
)
```

### Modular Design & Flexible Models

Switch between different embedding and LLM providers easily:

```python
from src.embedding import Embedder
from src.generation import Generator

# Use different embedding models
embedder = Embedder(provider="openai", model_name="text-embedding-3-large")
# embedder = Embedder(provider="gemini", model_name="models/embedding-001")
# embedder = Embedder(provider="qwen", model_name="text-embedding-v2")

# Use different LLMs
generator = Generator(provider="anthropic", model="claude-3-opus-20240229")
# generator = Generator(provider="gemini", model="gemini-pro")

# Or via config file
pipeline = RAGPipeline()  # Reads from config/config.yaml
```

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Monitoring and Logging

Logs are stored in `logs/rag_pipeline.log`. Configure logging level in `config.yaml`:

```yaml
logging:
  level: INFO # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: ./logs/rag_pipeline.log
```

Check vector store statistics:

```python
stats = pipeline.get_stats()
print(f"Total documents: {stats['document_count']}")
```

## 🔧 Advanced Usage

### Custom Prompts & Models

```python
from src.generation import PromptTemplate, Generator
from src.embedding import Embedder

# Custom prompt
custom_prompt = PromptTemplate(
    template="""Answer based on context:

Context: {context}
Question: {question}

Provide a detailed answer with citations."""
)

# Use different models
embedder = Embedder(provider="openai", model_name="text-embedding-3-large")
generator = Generator(provider="anthropic", model="claude-3-opus-20240229")

# Apply to pipeline
pipeline.embedder = embedder
pipeline.generator = generator

response = pipeline.query("What is the main topic?", custom_prompt=custom_prompt)
```

### Batch Processing

```python
queries = [
    "What is machine learning?",
    "Explain neural networks"
]

context_docs = [
    pipeline.retriever.retrieve(q)
    for q in queries
]

responses = pipeline.generator.batch_generate(
    queries=queries,
    context_documents_list=context_docs
)
```

## 🛠️ Troubleshooting

### ChromaDB Issues

If you encounter ChromaDB errors:

```bash
# Reset the vector store
python scripts/reset_vectorstore.py --confirm
```

### Memory Issues

For large datasets, adjust batch sizes in `config.yaml`:

```yaml
embedding:
  batch_size: 16 # Reduce if memory issues

ingestion:
  batch_size: 5
  max_workers: 2
```

## 📖 Documentation

- **Notebooks**: See `notebooks/` for interactive tutorials
- **Examples**: Check `examples/` for code samples
- **API Docs**: Each module has comprehensive docstrings

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for document processing
- [Milvus](https://milvus.io/) for high-performance vector storage
- [ChromaDB](https://www.trychroma.com/) for alternative vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy RAG-ing! 🚀**
