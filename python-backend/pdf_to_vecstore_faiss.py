from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


import os
# import numpy as np
# import faiss
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv



# Load api key variables from .env file into the environment
load_dotenv()



#Hardcoded file sources. Will need to be maitained separately outside of code in the future
# May need to be maintained in a database

pdfsource=[
            "https://www.singtel.com/content/dam/singtel/personal/products-services/tv/plupdates/SingtelTV_PL_Updates_SingtelTV_FAQs.pdf"
            ,"https://cdn2.singteldigital.com/content/dam/singtel/personal/products-services/tv/apps/tv-go/tv-go-documents/singteltvgo-faqs.pdf"
            # "https://www.singtel.com/content/dam/singtel/personal/products-services/mobile/info/bringitback/bring-it-back-tncs-samsung-s24-170124.pdf"
                ]


#---------------------
# FUNCTION TO CREATE PERSISTENT VECTORDB using FAISS
# This is to avoid vectorizing whole document again and getting charged if the embedding model is not free
#---------------------
def create_vectordb_from_pdfs(pdf_folder, persist_directory, embedmodel ="text-embedding-3-small"):
    # 1. Load PDFs from a directory
    def load_pdf(pdf_folder):
        documents = []
        for pdf_file in pdf_folder:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        
        return documents
    
    # 2. Split documents into manageable chunks
       # Will need to look on how to fine-tune chunking for better RAG results 
       # May need to apply additional cleaning to take out unnecessary contents footers other elements
       #  
    def split_text(documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        data = text_splitter.split_documents(documents)

        return data

    # prepare for docs for vectorization by running 2 functions above
    
    docs = split_text(load_pdf(pdf_folder))

    #3 vector store 
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(model=embedmodel))
    vectorstore.save_local(f"{persist_directory}/faiss_store")  # Persist to disk
   
#---------------------
# FUNCTION TO QUERY FROM SAVED PERSISTENT VECTORDB using FAISS
# This is to avoid vectorizing whole document again and getting charged if the embedding model is not free
#---------------------
def connect_and_query_vectordb(query, persist_directory, k=2,embedmodel ="text-embedding-3-small"):

    # 1. Load the persistent faiss index
    embeddings = OpenAIEmbeddings(model=embedmodel)
    vectorstore = FAISS.load_local(f"{persist_directory}/faiss_store", embeddings, allow_dangerous_deserialization=True)
    
    # retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    output = []
    for doc in docs:
        item = {
            "page_content": getattr(doc, "page_content", ""),
            "source": doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
        }
        output.append(item)
    return output



#------------------------------------------
# MANUAL LOADING OF FILES TO VECTOR STORE during setup 
# Will need to add routine when new files need to be updated into the DB
# -----------------------------------------

def loadpdf_to_vecdb(pdfsource):
    if os.path.isdir("./VectorDB"):
        print("DB already exist.. no new vectordb created")
        # pass
    else:
        print("new vectordb created")
        create_vectordb_from_pdfs(pdfsource,persist_directory = "./VectorDB",embedmodel="text-embedding-3-small")
        
#------------------------------------------
# MANUAL Testing functions for semantic search
# -----------------------------------------
def test_search(query):

    context = connect_and_query_vectordb(query, persist_directory = "./VectorDB",k=1,embedmodel ="text-embedding-3-small")
    
    return context


#------------------------------------------
# Inmemory Vectorstore functions
# -----------------------------------------
def Inmemory_RAG_ingest():
    pdfsource=["https://www.singtel.com/content/dam/singtel/personal/products-services/tv/plupdates/SingtelTV_PL_Updates_SingtelTV_FAQs.pdf"]
     
    # 1. Load PDFs from a directory
    def load_pdf(pdfsource):
        documents = []
        for pdf_file in pdfsource:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        
        return documents
    
    # 2. Split documents into manageable chunks
       # Will need to look on how to fine-tune chunking for better RAG results 
       #May need to apply additional cleaning to take out unnecessary contents footers other elements
       #  
    def split_text(documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        data = text_splitter.split_documents(documents)

        return data
    
        # prepare for docs for vectorization by running 2 functions above
    
    documents = split_text(load_pdf(pdfsource))
        # embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") #text-embedding-3-small  text-embedding-3-large
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)

    return vector_store
    


def Inmemory_RAG_Sourcing(query,vector_store): # AVOID USING TO PREVENT embedding charges

    results = vector_store.similarity_search(query, k=1)
    output = []
    for doc in results:
        item = {
            "page_content": getattr(doc, "page_content", ""),
            "source": doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
        }
        output.append(item)
    return output
    


if __name__ == '__main__':
    #load to a vector db in subfolder VectorDB (If there is none yet - will do one time loading for now and will not add new docs):
    loadpdf_to_vecdb(pdfsource) # load pdf files to vecDB if not yet loaded
   