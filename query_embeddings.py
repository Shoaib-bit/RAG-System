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
    vectordb = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
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
            # Prepare references for the prompt
            references = []
            context_parts = []
            
            for i, doc in enumerate(results):
                source_file = doc.metadata.get('source_file', 'Unknown')
                page_num = doc.metadata.get('page', 'N/A')
                ref_id = f"[{i+1}]"
                references.append(f"{ref_id} -> Document: {source_file}, Page: {page_num}, Chunk: {i+1}")
                context_parts.append(f"{ref_id}\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            references_text = "\n".join(references)

            prompt = f"""
            Based on the following information from the documents:

            {context}

            Please answer this question: {question}

            Follow these formatting rules in your answer:
            1. When referencing information, use numbered citations like [1], [2], [3], etc.
            2. After your answer, include a "References:" section that lists all sources you cited.
            3. Make sure all citations in your answer correspond to entries in the references section.
            4. Format each reference like this: [number] -> Document: filename, Page: number, Chunk: number

            Here are the references you should use:
            {references_text}
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
