import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from embedding import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"

logging.basicConfig(level=logging.INFO)

def create_chroma_instance():
    embedding_function = get_embedding_function()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    logging.info(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

def add_to_chroma(chunks: list[Document]):
    """Adds document chunks (text) to the Chroma database with persistence."""
    # Create the Chroma instance here for a robust approach
    db = create_chroma_instance()
    db.add_documents(chunks)
    # Persist the updated Chroma database state
    db.persist()
    logging.info("Added chunks to Chroma and persisted the database.")

def main():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()
