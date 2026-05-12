# GraphRAG Library

## Overview

GraphRAG Library is a modular implementation of a **Graph-based Retrieval-Augmented Generation (GraphRAG)** pipeline designed to enhance Large Language Model (LLM) reasoning using structured knowledge graphs and semantic retrieval.

This project combines:

- Knowledge Graph Construction
- Vector Embeddings
- Semantic Search
- Entity Relationship Mapping
- Context-Aware Question Answering
- LLM-Powered Reasoning

The goal of this repository is to provide a practical framework for building intelligent retrieval systems capable of understanding relationships between entities instead of relying only on traditional vector similarity search.

---

## What is GraphRAG?

Traditional RAG systems retrieve chunks of text using vector similarity.

GraphRAG improves this process by introducing **graph structures** that model entities and their relationships.

Instead of retrieving isolated text passages, GraphRAG enables:

- Multi-hop reasoning
- Relationship-aware retrieval
- Better contextual understanding
- Improved answer grounding
- Structured knowledge exploration

---

## Features

- Knowledge graph generation from documents
- Entity and relationship extraction
- Semantic vector search
- Hybrid retrieval pipeline
- Context-aware response generation
- Modular and extensible architecture
- Easy integration with LLMs
- Educational implementation for learning GraphRAG systems

---

## Project Architecture

```text
Documents
    │
    ▼
Text Processing & Chunking
    │
    ▼
Entity Extraction
    │
    ▼
Knowledge Graph Construction
    │
    ▼
Vector Embedding Generation
    │
    ▼
Hybrid Retrieval
(Graph + Vector Search)
    │
    ▼
LLM Response Generation
```

---

## Technologies Used

- Python
- Neo4j
- FAISS 
- Hugging Face Transformers
- Sentence Transformers
- NumPy

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MariamMoselhi/GraphRAG_Library.git
cd GraphRAG_Library
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main pipeline:

```bash
python main.py
```

Example workflow:

1. Load documents
2. Split text into chunks
3. Extract entities and relationships
4. Build knowledge graph
5. Generate embeddings
6. Perform hybrid retrieval
7. Generate LLM response

---

## Example Use Cases

- Intelligent document question answering
- Research paper exploration
- Medical knowledge assistants
- Enterprise knowledge management
- Educational AI systems
- Legal and compliance search
- Multi-document reasoning systems

---

## Key Concepts

### 1. Retrieval-Augmented Generation (RAG)

RAG combines external knowledge retrieval with language generation to improve factual accuracy and contextual relevance.

### 2. Knowledge Graphs

Knowledge graphs represent information as entities and relationships, enabling structured reasoning and semantic understanding.

### 3. Hybrid Retrieval

This project combines:

- Dense vector retrieval
- Graph traversal
- Entity-aware context expansion

to achieve more accurate and explainable responses.

---

## Educational Objectives

This repository is designed to help learners understand:

- Modern RAG architectures
- Knowledge graph construction
- Vector databases
- LLM orchestration
- Semantic retrieval systems
- Hybrid AI pipelines

---

## References

- Microsoft GraphRAG Documentation  
  https://microsoft.github.io/graphrag/

- Microsoft GraphRAG GitHub Repository  
  https://github.com/microsoft/graphrag

---

## License

This project is licensed under the MIT License.

---

## Author

### Mariam Moselhi
