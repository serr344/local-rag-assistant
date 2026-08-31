# Local RAG Assistant

A lightweight local Retrieval-Augmented Generation (RAG) application built with Microsoft Foundry Local.

The assistant answers questions using local documents only. Document embeddings are stored in SQLite, relevant chunks are retrieved with cosine similarity, and the retrieved context is passed to a local language model.

## Features

- Fully local inference with Microsoft Foundry Local
- PDF, DOCX, TXT, and Markdown support
- Automatic chunking and embedding
- SQLite-based local storage
- Cosine-similarity retrieval
- Relevance threshold for unrelated questions
- Incremental indexing for added, updated, or deleted documents
- Streamlit chat interface
- Source file, chunk, and similarity information
- Document upload and removal directly from the UI

## Project Structure

```text
local-rag-simple/
├── config.py
├── ingest.py
├── rag.py
├── main.py
├── ui.py
├── requirements.txt
├── setup.bat
├── run.bat
├── run_ui.bat
├── docs/
└── data/
```

## Setup

Run once:

```powershell
.\setup.bat
```

If Streamlit is not installed:

```powershell
.\.venv\Scripts\Activate.ps1
py -m pip install streamlit
```

## Run

Start the web interface:

```powershell
.\run_ui.bat
```

Or use the command-line interface:

```powershell
.\run.bat
```


## Example

If a history document about the foundation of the Republic of Turkey is placed in the `docs/` folder, you can ask:

```text
On what date did the Grand National Assembly of Turkey open?
```

Expected answer:

```text
23 April 1920
```

## How It Works

```text
Documents
   ↓
Chunking
   ↓
Embedding Model
   ↓
SQLite
   ↓
Query Embedding
   ↓
Cosine Similarity
   ↓
Top Relevant Chunks
   ↓
Local LLM
   ↓
Grounded Answer
```

Document changes are indexed automatically without reloading the local models.

## Supported Formats

- PDF
- DOCX
- TXT
- MD

## Limitations

- Scanned or image-only PDFs require OCR and are not supported directly.
- Retrieval quality depends on document content, chunking, and the relevance threshold.
- The current design targets small local document collections rather than large-scale vector databases.
