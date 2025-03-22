from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings  # 更新导入路径
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
import os
from .config import settings

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        self.vector_store = None
        self.llm = OpenAI(openai_api_key=settings.OPENAI_API_KEY) if settings.LLM_MODEL == "openai" else None
        
    def load_documents(self, file_path: str):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        self.vector_store = FAISS.from_documents(pages, self.embeddings)
        
    def query(self, question: str, k=3) -> str:
        if not self.vector_store:
            return "请先上传文档"
            
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n".join([d.page_content for d in docs])
        
        prompt = f"""基于以下上下文回答问题：
        {context}

        问题：{question}
        答案："""
        
        return self.llm.invoke(prompt) if self.llm else "本地模型响应示例"