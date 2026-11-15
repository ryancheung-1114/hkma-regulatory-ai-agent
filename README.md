# HKMA Regulatory Knowledge Agent

### AI-Powered Regulatory Intelligence for Hong Kong Banking

## Project Overview

An AI-powered regulatory knowledge assistant designed to support the interpretation, synthesis and practical application of Hong Kong Monetary Authority (HKMA) supervisory requirements.

The project addresses a common challenge in banking risk management: regulatory requirements are distributed across multiple supervisory documents, making it time-consuming to identify relevant provisions and understand their implications for credit risk management.

Using Retrieval-Augmented Generation (RAG), the application retrieves relevant regulatory evidence and generates integrated, source-grounded analyses.

## Key Features

- **Hybrid Retrieval:** Combines Qwen semantic embeddings with BM25 keyword search.
- **Reciprocal Rank Fusion (RRF):** Integrates semantic and keyword retrieval rankings.
- **AI Reranking:** Uses Qwen reranking to prioritize relevant regulatory passages.
- **Multi-step Regulatory Analysis:** Uses DeepSeek to decompose questions, assess evidence coverage and synthesize responses.
- **Traceable Citations:** Links generated answers to retrieved HKMA supervisory sources.
- **Incremental Indexing:** Reuses existing document embeddings to reduce processing costs.
- **Interactive Interface:** Streamlit application with retrieval diagnostics and source inspection.

## Regulatory Coverage

The current knowledge base covers 16 HKMA supervisory documents and circulars, including topics related to:

- Credit risk governance and approval
- Credit administration and portfolio monitoring
- Problem credit management
- Credit card operations
- Loan classification and provisioning
- Consumer credit data
- Stress testing and personal lending

## Technology Stack

| Component | Technology |
|---|---|
| Application | Python, Streamlit |
| Document Processing | PyMuPDF, LangChain |
| Semantic Retrieval | Qwen text-embedding-v4 |
| Keyword Retrieval | BM25 |
| Rank Fusion | Reciprocal Rank Fusion |
| Reranking | Qwen3-Rerank |
| Language Model | DeepSeek |
| Storage | Local JSON vector index |

## Example Use Case

**Question:**

How should a Hong Kong bank respond to increasing credit card delinquency when its NPL ratio remains stable?

**Analysis Objectives:**

1. Identify relevant HKMA supervisory requirements.
2. Connect credit monitoring, problem credit management and loan classification.
3. Assess potential early-warning indicators.
4. Develop practical credit risk management recommendations.
5. Distinguish regulatory obligations from professional analytical inferences.

## Project Purpose

Developed as an independent applied AI project combining retail credit risk management expertise with modern NLP, retrieval and reasoning techniques.

The project explores how generative AI can improve regulatory knowledge accessibility, analytical efficiency and evidence traceability in banking risk management.

## Limitations

This application is a research and demonstration project. AI-generated interpretations require human verification against applicable and current HKMA supervisory materials, particularly for compliance decisions.
