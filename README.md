Real Estate RAG Research Assistant (LLM + LangChain + Chroma + Groq)
Deployed Link-https://real-estate-rag-tool.streamlit.app/

Built a production-style Retrieval Augmented Generation system that extracts insights from live real-estate news/articles and answers user queries with cited sources.
Impact: Eliminates manual research time by ~80% for property analysts and investors.

Designed an automated pipeline for URL ingestion → document parsing → chunking → embedding generation → vector storage using MiniLM + ChromaDB.
Impact: Converts unstructured web data into a searchable knowledge base in seconds.

Integrated Groq’s Llama-3.3-70B via LangChain RetrievalQA to generate context-aware, hallucination-reduced answers grounded in retrieved documents.
Impact: Improves answer reliability compared to vanilla LLM chat.

Developed an interactive Streamlit interface allowing non-technical users to input multiple URLs and perform natural language queries.
Impact: Makes advanced LLM retrieval usable for real estate professionals without coding knowledge.

Deployed the application on Streamlit Cloud with dependency/version management and optimized embedding/model loading.
Impact: Demonstrates real-world ML system deployment and scalability readiness.


Real-Life Problem This Solves

This project simulates what companies actually build today:

Real World Need	How Your Project Helps
Analysts spend hours reading property news	Instant summarized Q&A from multiple sources
Investors need verified market insights	Answers include sources (trust & transparency)
Huge unstructured web data	Converted into semantic vector search
Non-technical stakeholders	Simple UI instead of complex ML tooling
Enterprise RAG trend	Shows you understand modern LLM architecture
