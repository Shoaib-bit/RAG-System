# RAG System

A Retrieval-Augmented Generation (RAG) system implementation in Python supporting multiple LLM providers: Google Gemini, OpenAI, and Anthropic Claude.

## Overview

This project implements a powerful RAG (Retrieval-Augmented Generation) system that enhances large language model responses by retrieving relevant context from your documents. It combines efficient document retrieval with various LLM providers to provide accurate, context-aware responses with citations.

## Features

- **Multiple LLM Provider Support**: Use Google Gemini, OpenAI, or Anthropic Claude models
- **Document Processing**: Ingest PDF documents and split them into manageable chunks
- **Vector-based Retrieval**: Utilize FAISS for efficient similarity search
- **Embeddings Generation**: Google's embedding model for high-quality vector representations
- **Context-aware Response Generation**: Enhance responses with relevant document context
- **Citation Support**: Responses include page and chunk citations
- **Interactive Query Interface**: Simple command-line interface for asking questions

## Technical Stack

- **Language Models**:
  - Google Gemini 2.0 Flash
  - OpenAI GPT-4o
  - Anthropic Claude 3 Opus
- **Embeddings**: Google Generative AI Embeddings (models/embedding-001)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: LangChain's document loaders and text splitters
- **Environment Management**: Python dotenv for API key management

## Installation

First, create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the `.env.example` file to `.env`:

   ```bash
   cp .env.example .env
   ```

2. Add your API keys to the `.env` file:

   ```
   # Google API Key for Gemini
   GOOGLE_API_KEY=your_google_api_key_here

   # OpenAI API Key
   OPENAI_API_KEY=your_openai_api_key_here

   # Anthropic API Key for Claude
   ANTHROPIC_API_KEY=your_anthropic_api_key_here

   # Default LLM provider (google, openai, or anthropic)
   LLM_PROVIDER=google

   # Default models for each provider
   GOOGLE_MODEL=gemini-2.0-flash
   OPENAI_MODEL=gpt-4o
   ANTHROPIC_MODEL=claude-3-opus-20240229
   ```

## Usage

The system has two main scripts:

1. **Create Embeddings**: Process documents and generate vector embeddings

   ```bash
   python create_embeddings.py --input docs --output faiss_index
   ```

   - `--input`: Path to directory containing documents or to a single document file (default: "docs")
   - `--output`: Directory to save the embeddings (default: "faiss_index")

2. **Query Embeddings**: Ask questions about your documents
   ```bash
   python query_embeddings.py --index faiss_index [--expanded] [--mmr] [--rerank] [--rerank_model MODEL_NAME] [--provider PROVIDER] [--model MODEL_NAME] [--compression] [--retrieval-qa]
   ```
   - `--index`: Directory containing the embeddings (default: "faiss_index")
   - `--expanded`: Enable expanded mode to generate more comprehensive answers by creating additional related queries
   - `--mmr`: Use Maximum Marginal Relevance search for more diverse results with reduced redundancy
   - `--rerank`: Enable CrossEncoder reranking for improved result relevance
   - `--rerank_model`: Specify which CrossEncoder model to use (default: "cross-encoder/ms-marco-MiniLM-L-6-v2")
   - `--provider`: Specify which LLM provider to use: "google", "openai", or "anthropic" (overrides .env setting)
   - `--model`: Specify a specific model to use with the chosen provider (overrides .env setting)
   - `--compression`: Use document compression to extract the most relevant content from retrieved documents
   - `--retrieval-qa`: Use LangChain's RetrievalQA chain for structured question answering with detailed references
   - Type 'exit' to quit the application

## Project Structure

```
create_embeddings.py    # Script to process documents and create embeddings
query_embeddings.py     # Script to query the embeddings and get answers
requirements.txt        # Python dependencies
docs/                   # Place PDF documents here
faiss_index/            # Vector database storage
  ├── index.faiss       # FAISS index file
  └── index.pkl         # Pickle file for document metadata
manage.py               # Deprecated main application file
.env.example            # Example environment file with configuration options
```

## How It Works

1. **Document Ingestion**: PDF documents are loaded and split into chunks
2. **Vector Embedding**: Text chunks are converted to vector embeddings
3. **Storage**: Embeddings are stored in a FAISS index for efficient retrieval
4. **Query Processing**: User questions are converted to embeddings and used for similarity search
5. **Response Generation**: Retrieved relevant context is sent to the chosen LLM with the user query
6. **Output**: The system returns context-enriched responses with citations

## Contributing

Feel free to open issues and pull requests. Contributions are welcome!

## License

MIT License. See LICENSE file for details.
