*RAG Notes Chatbot
This project is a simple Retrieval-Augmented Generation (RAG) chatbot built to understand how document-based question answering works in practice.
Instead of generating answers from scratch, the chatbot searches through a custom knowledge file and returns the most relevant information using embeddings and vector search.
The focus of this project is learning the core RAG concepts rather than building a complex UI.

*What this project does
-Reads a text file as a knowledge source
-Breaks the content into small, meaningful chunks
-Converts those chunks into vector embeddings
-Stores them in a vector database
-Retrieves the most relevant chunks based on a user’s question
-Displays results in a simple Streamlit web interface

*Tech stack
-Python
-Streamlit
-ChromaDB (vector database)
-LangChain community utilities
-sentence-transformers (local embeddings)

*Project structure
LLM Project/
├── app.py          # Streamlit chatbot interface
├── ingest.py       # Builds vector database from notes
├── notes.txt       # Knowledge base
├── requirements.txt
├── .gitignore
└── README.md

*How to run locally
-Install dependencies
-pip install -r requirements.txt
-Build the vector database
-python ingest.py

*Run the chatbot
python -m streamlit run app.py
The app will open at http://localhost:8501.

*Example questions
-What is the difference between ETL and ELT?
-Explain SCD Type 2
-How do indexes improve SQL performance?
-What is a data lake versus a data warehouse?

*What I learned from this project
-How embeddings represent semantic meaning
-How vector databases enable document search
-How RAG reduces hallucinations by grounding responses in data
-How to structure and version an end-to-end AI project
-Git and GitHub best practices

*Future improvements
-Support PDF uploads
-Add LLM-generated answers on top of retrieval
-Show source citations for each response
-Deploy the app online
