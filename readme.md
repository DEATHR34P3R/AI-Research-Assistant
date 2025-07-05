# AI Research Assistant

## Overview

This tool is built using **Streamlit** and acts as a research assistant that allows users to upload research papers or academic documents.

Supported file types:

- **PDF**
- **TXT** (only UTF-8 / UTF-16 encoded)

**Note:** It works best with digitally native or machine-readable PDFs — not scanned images.

## Features

The app currently includes three major functions:

1. **Ask Questions**\
   Ask free-form questions based on the document’s contents.

2. **Challenge Me**\
   Generates reasoning-based questions to test your understanding of the document.

3. **Summary**\
   Automatically generates a \~150-word summary of the document (customizable in code via `utils_1`).

## Under the Hood

- **LLM:** `Mistral-7B-Instruct-v0.1` by mistralai\
  Accessed via the **Together API**, but can be swapped for a local Ollama instance with minor edits in the `utils_1` files.

- **Embeddings:** `all-MiniLM-L6-v2` from HuggingFace

- **Vector Store:** FAISS (Facebook AI Similarity Search)

## Setup Instructions

```bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant

pip install -r requirements.txt
```

Add your Together API key in:

```
.streamlit/secrets.toml
```

```toml
together_api_key = "your-real-key-here"
```

Then run the app:

```bash
streamlit run main.py
```

## File Structure

```
ai-research-assistant/
├── main.py                         # Main Streamlit app
├── requirements.txt                # All dependencies
├── README.md                       # This file
├── .gitignore                      # Ignore secrets, virtual envs, etc.

├── utils_1/
│   ├── __init__.py
│   ├── doc_loader.py               # Loads and chunks text
│   ├── qa_handler.py               # Handles question answering
│   ├── Summarizer.py               # Summarizes large documents
│   └── challenge_me.py             # Generates and evaluates quiz questions

├── .streamlit/
│   ├── config.toml                 # Optional UI settings
│   └── secrets.toml                # API keys (ignored by Git)
```

## Architecture

```
            Upload Document
                    |
                    v
         Extract + Chunk Text  (doc_loader.py)
                    |
                    v
     Embed Chunks into Vectors (FAISS + HuggingFace)
                    |
                    v
     Semantic Search for Context (similarity_search)
                    |
                    v
           Answer with LLM (qa_handler.py)

                  OR

        Generate Reasoning Questions (challenge_me.py)
                    |
                    v
         User Answers + AI Feedback

                  OR

           Generate Final Summary (Summarizer.py)
```

