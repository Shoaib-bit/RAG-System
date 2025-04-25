from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # FAISS instead of Chroma

import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001")

loader = PyPDFLoader("docs/large-doc.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

vectordb = FAISS.from_documents(texts, embedding=embeddings)

vectordb.save_local("faiss_index")


while True:
    question = input("\nAsk a question: ")
    if question.lower() == "exit":
        break

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