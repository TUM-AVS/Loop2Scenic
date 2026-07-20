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

- **Python 3.12.3** (recommended; tested with conda)
- **Conda** (Miniconda or Anaconda)
- **Docker** and **Docker Compose** (for Milvus)
- **CUDA-capable GPU** (recommended for Qwen embedding/reranking models)
- **Linux** (required for CARLA simulation)
- **Git**



### Environment Setup

Follow these steps in order.

#### 1. Clone the repository

```bash
git clone https://github.com/CelanLi/ads-mrag.git
cd ads-mrag
```

If the repository uses submodules, initialize them:

```bash
git submodule update --init --recursive
```



#### 2. Create a Conda environment (Python 3.12.3)

```bash
conda create -n ads-mrag python=3.12.3 -y
conda activate ads-mrag
python --version  # should print Python 3.12.3
```



#### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```



#### 4. Start Milvus and configure environment variables

Start the vector database:

```bash
docker-compose up -d
docker-compose ps
curl http://localhost:9091/healthz
```

Copy and edit environment variables:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys as needed:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `QWEN_API_KEY`
- `DEEPSEEK_API_KEY`

> For detailed Milvus setup, troubleshooting, and configuration options, see [MILVUS_SETUP.md](MILVUS_SETUP.md).



#### 5. Download embedding models from Hugging Face

Local Qwen VL models used by the pipeline should be downloaded into `./models`.
Install the Hugging Face CLI if needed:

```bash
pip install -U huggingface_hub
```

Example: download **Qwen3-VL-Embedding-2B**:

```bash
hf download Qwen/Qwen3-VL-Embedding-2B \
  --local-dir ./models/Qwen3-VL-Embedding-2B
```

> **Note:** The Hugging Face CLI command is now `hf` (replacing the older `huggingface-cli`).

Then point `config/config.yaml` to the local path:

```yaml
embedding:
  provider: qwen
  model_name: Qwen3-VL-Embedding-2B
  model_path: ./models/Qwen3-VL-Embedding-2B
```

Download other models the same way (for example reranker models) and update the corresponding paths in `config/config.yaml`.

#### 6. Install CARLA 0.9.16 and the Python API (Python 3.12.3)

CARLA is the driving simulator used for Scenic batch runs. Install it **next to** the project directory (one level above the repo root):

```
chenli2/
├── ads-mrag/          ← this repository
└── carla_0.9.16/      ← CARLA simulator (you create this)
```

**6.1 Download and extract the CARLA server**

From the parent directory of the repo, download and extract the pre-built Linux binaries:

```bash
cd ..   # from ads-mrag/ to the parent folder
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz
tar -xzf CARLA_0.9.16.tar.gz
```

The archive may extract to `CARLA_0.9.16/`. If you prefer lowercase, rename it:

```bash
mv CARLA_0.9.16 carla_0.9.16
```

After extraction, you should have `CarlaUE4.sh` at `../carla_0.9.16/CarlaUE4.sh`.

**6.2 Install the CARLA Python wheel**

Activate the Conda environment from step 2, then install the wheel that matches Python 3.12:

```bash
conda activate ads-mrag
cd ../carla_0.9.16/PythonAPI/carla/dist
pip install carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl
```

Verify the install:

```bash
python -c "import carla; print(carla.__file__)"
```

**6.3 Point the project config at CARLA**

Edit `config/config.yaml` and set:

- `simulation.carla.binary_dir` — absolute path to the CARLA folder (the directory that contains `CarlaUE4.sh`)
- `simulation.scenic.conda_env` — name of the Conda environment you created in step 2 (for example `ads-mrag`)

Example:

```yaml
simulation:
  carla:
    binary_dir: "/home/your_user/chenli2/carla_0.9.16"
  scenic:
    conda_env: ads-mrag
```

Replace `/home/your_user/chenli2/carla_0.9.16` with the actual path on your machine.



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