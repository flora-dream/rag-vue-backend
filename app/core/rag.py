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
        self.vector_store = self._load_existing_index()  # 初始化时尝试加载已有索引
        self.llm = OpenAI(openai_api_key=settings.OPENAI_API_KEY) if settings.LLM_MODEL == "openai" else None

    def _load_existing_index(self):
        """加载已存在的FAISS索引"""
        if os.path.exists(settings.FAISS_INDEX_PATH):
            print("检测到已有索引，正在加载...")
            return FAISS.load_local(
                folder_path=settings.FAISS_INDEX_PATH,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True  # 需要此参数来加载自定义类
            )
        return None
        
    def load_documents(self, file_path: str):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        # 如果已有索引则合并，否则新建
        if self.vector_store:
            print("合并到已有索引...")
            self.vector_store.add_documents(pages)
        else:
            print("创建新索引...")
            self.vector_store = FAISS.from_documents(pages, self.embeddings)
        
        # 持久化保存索引
        self.vector_store.save_local(settings.FAISS_INDEX_PATH)
        print(f"索引已保存到：{settings.FAISS_INDEX_PATH}")
        
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