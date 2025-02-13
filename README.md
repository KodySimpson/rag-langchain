## Table of Contents

| Section | File |
|---------|------|
| Environmental Variables | [env_variables.ipynb](basics/env_variables.ipynb) |
| Chat Models | [chat_models.ipynb](basics/chat_models.ipynb) |
| Using Ollama | [using_ollama.ipynb](basics/using_ollama.ipynb) |
| Document Loaders | [doc_loading.ipynb](indexing/doc_loading.ipynb) |
| Splitting | [splitting.ipynb](indexing/splitting.ipynb) |
| Embeddings | [embeddings.ipynb](indexing/embeddings.ipynb) |
| Vector Stores | [vector_dbs.ipynb](indexing/vector_dbs.ipynb) |
| Retrievers | [retrievers.ipynb](generation/retrievers.ipynb) |
| Full RAG Example | [full_rag.py](generation/full_rag.py) |
| Web RAG | [web_rag.py](apps/web_rag.py) |
| Web RAG Extended | [web_rag_extended.py](apps/web_rag_extended.py) |



## Setup Instructions

### 1. Prerequisites
- Python 3.11 or higher
- pip (Python package installer)

### 2. Create a Virtual Environment
First, create a new virtual environment in the project directory:
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
on Linux/Mac:
```bash
source venv/bin/activate
```

on Windows:
```bash
venv\Scripts\activate
```

### 4. Installing new packages
```bash
pip install package_name
```

if you want to install all the packages in the requirements.txt file, you can use the following command:
```bash
pip install -r requirements.txt
```

### 5. Saving the requirements.txt file
```bash
pip freeze > requirements.txt
```

