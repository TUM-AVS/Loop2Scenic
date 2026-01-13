"""
Setup script for ADS-MRAG package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="ads-mrag",
    version="0.1.0",
    description="Advanced Document Search - Metadata RAG Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/ads-mrag",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "chromadb>=0.4.22",
        "langchain>=0.1.0",
        "langchain-community>=0.0.13",
        "openai>=1.10.0",
        "tiktoken>=0.5.2",
        "pypdf>=3.17.4",
        "python-docx>=1.1.0",
        "markdown>=3.5.1",
        "beautifulsoup4>=4.12.2",
        "sentence-transformers>=2.2.2",
        "torch>=2.1.0",
        "pandas>=2.1.4",
        "numpy>=1.24.3",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.3",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.1",
        "tenacity>=8.2.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "api": [
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.28.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
