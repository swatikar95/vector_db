import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))




if __name__ == "__main__":
    print("Hello")
    file_path = "mediumblogs\\mediumblog1.txt"
    loader = TextLoader(file_path,encoding='utf-8')
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))
    
    embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))
    dosearch = PineconeVectorStore.from_documents(texts,embeddings,index_name="medium-blog")

    qa = RetrievalQA.from_chain_type(
        llm = OpenAI(),chain_type="stuff",retriever=dosearch.as_retriever(),return_source_documents=True
    )

    query = "What is vector db?Give me 15 words answer"
    result = qa({"query":query})
    print(result)