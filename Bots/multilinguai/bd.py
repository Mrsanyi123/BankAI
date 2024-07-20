from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embedding import get_embedding_function
from langchain.vectorstores.chroma import Chroma
import logging

logging.basicConfig(level=logging.INFO)

CHROMA_PATH = "chroma"
DATA_PATH = "data/MultinguAI_guide.pdf"


def main(): 
    # Create (or update) the data store.
    documents = load_documents()
    logging.info(f"Loaded {len(documents)} documents")
    chunks = split_documents(documents)
    logging.info(f"Split into {len(chunks)} chunks")
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    db.add_documents(chunks)
    db.persist()


if __name__ == "__main__":
    main()
