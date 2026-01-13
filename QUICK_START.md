# Quick Start Guide

Get started with ADS-MRAG in 5 minutes!

## Installation

### Step 1: Clone and Setup

```bash
# Clone the repository
cd ads-mrag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start Milvus

```bash
# Start Milvus using Docker (recommended)
docker-compose up -d

# Verify Milvus is running
docker-compose ps
```

**Alternative**: Use Milvus Lite (no Docker):

```bash
pip install milvus
```

> For detailed Milvus setup, see [MILVUS_SETUP.md](MILVUS_SETUP.md)

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_key_here
```

## Basic Usage

### Example 1: Quick Test with Sample Data

```python
from src.pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline()

# Add sample documents
texts = [
    "Python is great for data science and machine learning.",
    "JavaScript is commonly used for web development."
]
tags = [
    ["programming", "python", "data-science"],
    ["programming", "javascript", "web"]
]

chunks = pipeline.processor.process_texts(texts=texts, tags=tags)
pipeline.vectorstore.add_documents(chunks)

# Query with tag filtering
response = pipeline.query(
    query="Tell me about data science",
    filter_tags=["data-science"],
    return_sources=True
)

print(response["response"])
```

### Example 2: Ingest Documents from Files

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()

# Ingest a directory of documents
pipeline.ingest_documents(
    source="data/raw/",
    tags=["research", "2024"],
    is_directory=True
)

# Query
response = pipeline.query(
    "What are the key findings?",
    filter_tags=["research"]
)
print(response)
```

### Example 3: Command Line Usage

```bash
# Ingest documents
python scripts/ingest_documents.py \
    --source data/raw/sample.pdf \
    --tags "finance,report,Q1-2024"

# Query
python scripts/query_pipeline.py \
    --query "What were the Q1 results?" \
    --tags finance \
    --show-sources
```

## Understanding Tag-Based Filtering

Tags allow you to organize and filter your documents:

```python
# Add documents with different tags
pipeline.processor.process_texts(
    texts=[
        "Annual financial report for 2024...",
        "Technical documentation for API...",
        "Marketing strategy for Q2..."
    ],
    tags=[
        ["finance", "annual", "2024"],
        ["technical", "api", "documentation"],
        ["marketing", "strategy", "Q2"]
    ]
)

# Query only financial documents
response = pipeline.query(
    "What were the financial results?",
    filter_tags=["finance"]
)

# Query only 2024 documents
response = pipeline.query(
    "What happened in 2024?",
    filter_tags=["2024"]
)

# Documents with ANY of the specified tags (OR operation)
response = pipeline.query(
    "Show me recent updates",
    filter_tags=["finance", "marketing"]  # Returns docs with either tag
)
```

## Key Configuration Options

Edit `config/config.yaml` to customize:

```yaml
# Switch embedding provider
embedding:
  provider: openai # or huggingface
  model_name: text-embedding-3-large

# Switch LLM provider
llm:
  provider: anthropic # or openai
  model: claude-3-sonnet-20240229

# Adjust retrieval
retrieval:
  top_k: 10
  similarity_threshold: 0.6
```

## Common Use Cases

### 1. Knowledge Base Search

```python
# Ingest company documentation
pipeline.ingest_documents("docs/", tags=["company", "internal"], is_directory=True)

# Search with context
response = pipeline.query("How do I request vacation?", filter_tags=["company"])
```

### 2. Research Paper Analysis

```python
# Ingest papers by topic
pipeline.ingest_documents("papers/ml/", tags=["research", "ml"], is_directory=True)
pipeline.ingest_documents("papers/nlp/", tags=["research", "nlp"], is_directory=True)

# Query specific area
response = pipeline.query("What are recent advances in transformers?", filter_tags=["nlp"])
```

### 3. Customer Support

```python
# Organize by product and version
pipeline.ingest_documents("docs/product_a/", tags=["product-a", "v2.0"], is_directory=True)
pipeline.ingest_documents("docs/product_b/", tags=["product-b", "v1.5"], is_directory=True)

# Product-specific queries
response = pipeline.query("How to reset password?", filter_tags=["product-a"])
```

## Next Steps

- 📓 Check out the [Jupyter notebooks](notebooks/) for interactive examples
- 📚 Read the full [README.md](README.md) for detailed documentation
- 🧪 Run the tests: `pytest tests/ -v`
- 🔍 Explore the [examples](examples/) directory

## Troubleshooting

### Issue: "No OpenAI API key found"

**Solution**: Make sure you've set `OPENAI_API_KEY` in your `.env` file

### Issue: "ChromaDB errors"

**Solution**: Reset the vector store:

```bash
python scripts/reset_vectorstore.py --confirm
```

### Issue: "Out of memory"

**Solution**: Reduce batch sizes in `config/config.yaml`:

```yaml
embedding:
  batch_size: 16
ingestion:
  max_workers: 2
```

## Getting Help

- Check the [README.md](README.md) for comprehensive documentation
- Look at [examples/](examples/) for code samples
- Review [tests/](tests/) to see how components work

Happy RAG-ing! 🚀
