from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS

import os
import argparse

def query_embeddings(index_dir="faiss_index"):
    """Query embeddings and ask questions."""
    load_dotenv()

    # Initialize LLM and embeddings
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

    # Load the vector database
    print(f"Loading embeddings from: {index_dir}")
    vectordb = FAISS.load_local(index_dir, embeddings)
    print("Embeddings loaded successfully!")

    # Interactive query loop
    print("\nEnter 'exit' to quit the program")
    while True:
        question = input("\nAsk a question: ")
        if question.lower() == "exit":
            break

        # Search for relevant documents
        results = vectordb.similarity_search(question, k=3)

        if results:
            context = "\n".join([d.page_content for d in results])

            prompt = f"""
            Based on the following information from the document:
            include citations in the format [page number] and [chunk number]

            {context}

            Please answer this question: {question}
            """

            response = llm.invoke(prompt)

            print("\nAnswer:")
            print(response.content)

        else:
            print("\nSorry, no relevant information found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query document embeddings")
    parser.add_argument("--index", type=str, default="faiss_index",
                        help="Directory containing the embeddings")

    args = parser.parse_args()
    query_embeddings(args.index)
