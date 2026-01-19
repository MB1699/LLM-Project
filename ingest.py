import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def main():
    # ✅ Change this depending on where your notes file is:
    # If your notes is in the root: "notes.txt"
    # If it's in data folder: "data/notes.txt"
    path = "notes.txt"   # <-- set this to your real location

    print("DEBUG cwd:", os.getcwd())
    print("DEBUG looking for:", os.path.abspath(path))

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {os.path.abspath(path)}")

    # check file size
    size = os.path.getsize(path)
    print("DEBUG notes.txt bytes:", size)
    if size == 0:
        raise ValueError("notes.txt is empty (0 bytes). Add content and try again.")

    # load
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    print("DEBUG docs loaded:", len(docs))

    if len(docs) == 0 or len(docs[0].page_content.strip()) == 0:
        raise ValueError("TextLoader loaded 0 content. Check encoding or file content.")

    print("DEBUG first 200 chars:\n", docs[0].page_content[:200])

    # split
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print("DEBUG chunks created:", len(chunks))

    # show chunk sizes
    print("DEBUG chunk lengths (first 5):", [len(c.page_content) for c in chunks[:5]])

    if len(chunks) == 0:
        raise ValueError("No chunks created. Something is wrong with the input text.")

    # IMPORTANT: delete old DB if it exists (prevents weird states)
    # We'll just reuse directory, but you can delete manually too.
    db = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory="chroma_db"
    )
    db.persist()
    print(f"✅ Stored {len(chunks)} chunks into chroma_db")

if __name__ == "__main__":
    main()
