from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
import argparse

def get_llm(provider=None, model=None):
    """Get LLM based on provider and model from config or arguments."""
    # If no provider specified, check env var, default to google
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "google").lower()

    # Initialize the appropriate LLM based on provider
    if provider == "google":
        if not model:
            model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(model=model)
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        if not model:
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return ChatOpenAI(model=model)
    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY in your .env file.")
        if not model:
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        return ChatAnthropic(model=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers are 'google', 'openai', and 'anthropic'.")

def query_embeddings(index_dir="faiss_index", expanded=False, mmr=False, rerank=False,
                    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2", provider=None, model=None, 
                    use_compression=False, use_retrieval_qa=False):
    """Query embeddings and ask questions."""
    load_dotenv()

    # Initialize LLM and embeddings
    try:
        llm = get_llm(provider, model)
        # Get the actual model being used for display
        actual_provider = provider or os.getenv('LLM_PROVIDER', 'google')
        actual_model = model
        if not actual_model:
            if actual_provider == "google":
                actual_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            elif actual_provider == "openai":
                actual_model = os.getenv("OPENAI_MODEL", "gpt-4o")
            elif actual_provider == "anthropic":
                actual_model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
                
        print(f"Using {actual_provider} with model: {actual_model}")
    except ValueError as e:
        print(f"Error initializing LLM: {str(e)}")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

    # Initialize CrossEncoder for reranking if enabled
    cross_encoder = None
    if rerank:
        print(f"Loading CrossEncoder model: {rerank_model}")
        cross_encoder = CrossEncoder(rerank_model)
        print("CrossEncoder model loaded successfully!")

    # Load the vector database
    print(f"Loading embeddings from: {index_dir}")
    vectordb = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    print("Embeddings loaded successfully!")

    # Setup retriever based on options
    if mmr:
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Setup compression if enabled
    if use_compression:
        print("Using document compression to extract most relevant content...")
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        retriever = compression_retriever

    # Setup RetrievalQA chain if enabled
    retrieval_qa_chain = None
    if use_retrieval_qa:
        print("Setting up RetrievalQA chain...")
        prompt_template = """
        Answer the question based on the context below. 
        
        Context:
        {context}
        
        Question: {question}
        
        Please provide a detailed answer with citations to the source documents in the format [1], [2], etc.
        After your answer, include a "References:" section listing all cited sources.
        """
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    # Interactive query loop
    print("\nEnter 'exit' to quit the program")
    while True:
        question = input("\nAsk a question: ")
        if question.lower() == "exit":
            break

        if use_retrieval_qa:
            # Use RetrievalQA chain for answering
            result = retrieval_qa_chain.invoke({"query": question})
            
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Format references
            references = []
            for i, doc in enumerate(source_docs):
                source_file = doc.metadata.get('source_file', 'Unknown')
                page_num = doc.metadata.get('page', 'N/A')
                references.append(f"[{i+1}] -> Document: {source_file}, Page: {page_num}, Chunk: {i+1}")
            
            # Add references if not already included
            if "References:" not in answer:
                answer += "\n\nReferences:\n" + "\n".join(references)
            
            print("\nAnswer:")
            print(answer)
            continue
            
        # Search for relevant documents
        if expanded:
            print("Using expanded mode to generate more comprehensive answers...")
            # Generate related queries using LLM
            expansion_prompt = f"""
            Given the user question: "{question}"

            Please generate 3 related but more specific questions that would help provide a comprehensive answer.
            Return only the questions as a numbered list without any introduction or explanation.
            """

            expansion_response = llm.invoke(expansion_prompt)
            expanded_questions = [question] + [q.strip() for q in expansion_response.content.split('\n') if q.strip() and any(c.isdigit() for c in q[:2])]

            print(f"Generated {len(expanded_questions)-1} additional queries")

            # Collect results from all queries
            all_results = []
            seen_content = set()

            for q in expanded_questions:
                if use_compression:
                    q_results = retriever.invoke(q)
                elif mmr:
                    q_results = vectordb.max_marginal_relevance_search(q, k=3, fetch_k=10)
                else:
                    q_results = vectordb.similarity_search(q, k=3)
                for doc in q_results:
                    # Deduplicate results
                    if doc.page_content not in seen_content:
                        all_results.append(doc)
                        seen_content.add(doc.page_content)

            results = all_results
        else:
            if use_compression:
                print("Retrieving and compressing documents...")
                results = retriever.invoke(question)
            elif mmr:
                print("Using MMR search for maximum relevance and diversity...")
                results = vectordb.max_marginal_relevance_search(question, k=3, fetch_k=10)
            else:
                results = vectordb.similarity_search(question, k=3)

        if rerank and results:
            print("Reranking results using CrossEncoder...")
            rerank_inputs = [(question, doc.page_content) for doc in results]
            rerank_scores = cross_encoder.predict(rerank_inputs)
            results = [doc for _, doc in sorted(zip(rerank_scores, results), key=lambda x: x[0], reverse=True)]

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
    parser.add_argument("--expanded", action="store_true",
                        help="Use expanded mode to generate more comprehensive answers")
    parser.add_argument("--mmr", action="store_true",
                        help="Use Maximum Marginal Relevance for more diverse search results")
    parser.add_argument("--rerank", action="store_true",
                        help="Use CrossEncoder for reranking search results")
    parser.add_argument("--rerank_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="CrossEncoder model to use for reranking")
    parser.add_argument("--provider", type=str, choices=["google", "openai", "anthropic"],
                        help="LLM provider to use (google, openai, or anthropic)")
    parser.add_argument("--model", type=str,
                        help="Specific model to use with the chosen provider")
    parser.add_argument("--compression", action="store_true",
                        help="Use document compression to extract most relevant content")
    parser.add_argument("--retrieval-qa", action="store_true",
                        help="Use RetrievalQA chain for answering questions")

    args = parser.parse_args()
    query_embeddings(args.index, args.expanded, args.mmr, args.rerank, args.rerank_model,
                    args.provider, args.model, args.compression, args.retrieval_qa)
