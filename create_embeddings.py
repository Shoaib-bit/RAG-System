from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # FAISS instead of Chroma

import os
import argparse

def create_embeddings(pdf_path, output_dir="faiss_index"):
    """Create and save embeddings from a PDF document."""
    load_dotenv()

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

    # Load document
    print(f"Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")

    # Create vector database
    print("Creating vector embeddings...")
    vectordb = FAISS.from_documents(texts, embedding=embeddings)

    # Save embeddings
    print(f"Saving embeddings to: {output_dir}")
    vectordb.save_local(output_dir)
    print("Embeddings created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings from a PDF document")
    parser.add_argument("--pdf", type=str, default="docs/large-doc.pdf",
                        help="Path to the PDF document")
    parser.add_argument("--output", type=str, default="faiss_index",
                        help="Directory to save the embeddings")

    args = parser.parse_args()
    create_embeddings(args.pdf, args.output)
