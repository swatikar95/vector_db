from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI


load_dotenv()

if __name__ == "__main__":
    print("hello")
    file_path = "2210.03629.pdf"
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=20,separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    # print(len(docs))
    embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))
    vectorestore = FAISS.from_documents(docs,embeddings)
    vectorestore.save_local("fais_index_local")


    new_vectorestore = FAISS.load_local("fais_index_local",embeddings,allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff",retriever=new_vectorestore.as_retriever())
    res = qa.run("Give me a summary of REACT in three sentences")
    print(res)


