# 🚦 AcciEye | TALE Intelligence Module

**T**raffic **A**ccident **L**iability **E**ngine

This directory contains the **TALE Intelligence Module**, the legal reasoning core of the broader **AcciEye** platform. It acts as a deterministic decision-making interface that consumes visual state metadata from the AcciEye YOLOv26m object detection pipeline to calculate legal liability and generate formal traffic accident reports.

## 📖 Overview & Engineering Philosophy

Large Language Models (LLMs) are prone to hallucinating right-of-way in complex legal and spatial traffic scenarios. Traditional Retrieval-Augmented Generation (RAG) fails in this domain because traffic laws are strict, relational logic systems rather than just semantic text.

**TALE** overcomes this by implementing **Knowledge-Augmented Generation (KAG)**. The system receives accident metadata from the **AcciEye YOLOv26m** pipeline, routes the visual data through a hardcoded `NetworkX` graph to mathematically prove right-of-way, and synthesizes the final verdict using **Gemini 2.5 Flash** alongside flawlessly chunked bilingual legal data.

## ✨ Core Architecture (The 3-Phase Pipeline)

1. **Phase 1: Perception & State Extraction (YOLOv26m + VLM)**
* The module consumes processed object detection data from the **AcciEye YOLOv26m** pipeline.
* A Vision-Language Model (VLM) extracts vehicle environmental states (e.g., "Silver Sedan on Main Road") and outputs strict JSON.


2. **Phase 2: Deterministic Legal Graph (NetworkX)**
* To eliminate LLM hallucination, the state JSON is fed into a directed Knowledge Graph (`DiGraph`).
* This graph represents the unbreakable hierarchy of Jordanian Traffic Law (Traffic Officer > Lights > Signs > Road Markings > Default Rules) to calculate a mathematically proven fault verdict.


3. **Phase 3: Knowledge-Augmented Generation (KAG) & Synthesis**
* The system performs semantic retrieval against a local FAISS vector database.
* **Smart Data Ingestion:** Our custom Regex chunker preserves legal context based on Arabic-Indic numeric formatting (١. , ١٫١.).
* **Gemini 2.5 Flash** synthesizes the visual data, the deterministic graph verdict, and the localized legal text into a cohesive, formal liability report.



## 🛠️ Technology Stack

* **Computer Vision Integration:** YOLOv26m (via AcciEye pipeline)
* **Orchestration & LLM:** LangChain, OpenRouter API (**Gemini 2.5 Flash**)
* **Knowledge Graph:** NetworkX (Foundation for the KAG / GraphRAG logic)
* **Vector Database:** FAISS (Local In-Memory Index)
* **Embeddings:** HuggingFace `paraphrase-multilingual-MiniLM-L12-v2`
* **Data Parsing:** PDFPlumber (Custom regex semantic chunking)
* **Frontend Dashboard:** Streamlit (Unified AcciEye UI)

## 📂 Project Structure

```text
tale-module/
├── .env                        # API Keys (OpenRouter, HuggingFace)
├── requirements.txt            # Python dependencies
├── rag_ingest.py               # Utility to build the FAISS KAG database
├── data/
│   └── vectorstore/            # Compiled bilingual FAISS index
├── docs/
│   ├── PSD_Traffic_Manual_EN.pdf
│   └── PSD_Traffic_Manual_AR.pdf
└── app/
    ├── core/
    │   ├── config.py           # Paths and model selection
    │   ├── prompts.py          # Version-controlled system instructions
    │   └── graph_logic.py      # NetworkX Deterministic Right-of-Way Graph
    └── services/
        ├── rag_service.py      # PDF parsing and semantic retrieval
        └── agent_service.py    # Langchain orchestration (Gemini 2.5 Flash)

```

## 🚀 Installation & Setup

**1. Clone the repository and install dependencies:**

```bash
git clone https://github.com/your-org/AcciEye.git
cd AcciEye/tale-module
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

**2. Configure Environment Variables (`.env`):**

```env
OPENROUTER_API_KEY=your_openrouter_api_key
HF_TOKEN=your_huggingface_read_token

```

**3. Build the Legal KAG Database:**

```bash
python rag_ingest.py

```

## 🎓 TALE Developer

**[Oun Alawamleh]** - AI Engineer

