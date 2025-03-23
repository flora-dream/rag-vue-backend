from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from app.core.rag import RAGSystem
from app.core.config import settings
import os

router = APIRouter()
rag = RAGSystem()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "仅支持PDF文件")
    
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    file_path = os.path.join(settings.DATA_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        rag.load_documents(file_path)
        return {
            "message": "文件上传并加载成功",
            "index_size": rag.vector_store.index.ntotal  # 显示当前索引包含的文档数
        }
    except Exception as e:
        raise HTTPException(500, f"文件处理失败: {str(e)}")

@router.post("/ask")
async def ask_question(question: str = Body(..., embed=True)):
    if not question:
        raise HTTPException(400, "问题不能为空")
    return {"answer": rag.query(question)}