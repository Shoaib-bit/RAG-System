from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredRTFLoader,
    DirectoryLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # FAISS instead of Chroma

import os
import argparse
import glob
from typing import List, Dict, Any

def get_loader_mapping():
    """Return mapping of file extensions to document loaders."""
    return {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".rtf": UnstructuredRTFLoader,
        # Default loader for other file types
        "default": UnstructuredFileLoader
    }

def load_documents(directory_path: str) -> List[Any]:
    """Load documents recursively from directory."""
    print(f"Scanning directory: {directory_path}")

    loader_mapping = get_loader_mapping()
    all_documents = []

    # Walk through directory recursively
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)

            # Skip hidden files and unsupported formats
            if file.startswith('.'):
                continue

            try:
                # Select appropriate loader based on file extension
                if file_ext in loader_mapping:
                    loader_class = loader_mapping[file_ext]
                    print(f"Loading {file_ext} file: {file_path}")
                    loader = loader_class(file_path)
                    documents = loader.load()
                    
                    # Add source filename to metadata
                    for doc in documents:
                        if not hasattr(doc, 'metadata'):
                            doc.metadata = {}
                        doc.metadata['source_file'] = file_name
                    
                    all_documents.extend(documents)
                    print(f"Successfully loaded {len(documents)} document(s) from {file_path}")
                else:
                    # Try with default loader for unknown file types that might be supported
                    try:
                        print(f"Attempting to load unknown file type: {file_path}")
                        loader = loader_mapping["default"](file_path)
                        documents = loader.load()
                        
                        # Add source filename to metadata
                        for doc in documents:
                            if not hasattr(doc, 'metadata'):
                                doc.metadata = {}
                            doc.metadata['source_file'] = file_name
                        
                        all_documents.extend(documents)
                        print(f"Successfully loaded {len(documents)} document(s) from {file_path}")
                    except Exception as e:
                        print(f"Could not load {file_path}: {str(e)}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

    print(f"Loaded {len(all_documents)} total documents from {directory_path}")
    return all_documents

def create_embeddings(input_path, output_dir="faiss_index"):
    """Create and save embeddings from documents in a directory."""
    load_dotenv()

    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

    # Load documents
    if os.path.isdir(input_path):
        documents = load_documents(input_path)
    else:
        # If it's a single file, determine the loader based on extension
        file_ext = os.path.splitext(input_path)[1].lower()
        file_name = os.path.basename(input_path)
        loader_mapping = get_loader_mapping()

        if file_ext in loader_mapping:
            loader_class = loader_mapping[file_ext]
        else:
            loader_class = loader_mapping["default"]

        print(f"Loading single file: {input_path}")
        try:
            loader = loader_class(input_path)
            documents = loader.load()
            
            # Add source filename to metadata
            for doc in documents:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['source_file'] = file_name
                
            print(f"Successfully loaded {len(documents)} document(s)")
        except Exception as e:
            print(f"Error loading {input_path}: {str(e)}")
            return

    if not documents:
        print("No documents were loaded. Embeddings cannot be created.")
        return

    # Split text
    print("Splitting documents into chunks...")
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
    parser = argparse.ArgumentParser(description="Create embeddings from documents")
    parser.add_argument("--input", type=str, default="docs",
                        help="Path to directory containing documents or to a single document file")
    parser.add_argument("--output", type=str, default="faiss_index",
                        help="Directory to save the embeddings")

    args = parser.parse_args()
    create_embeddings(args.input, args.output)
