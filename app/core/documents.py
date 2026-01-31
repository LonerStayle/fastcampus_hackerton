from typing import  List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from functools import lru_cache
import os

@lru_cache(maxsize=1)
def get_documents() -> List[Document]:
    """
    RAG에서 사용하는 모든 원본 문서를 로드한다.
    - BM25
    - VectorStore 인덱싱
    - Retriever
    공통 소스
    """
# 현재 스크립트의 디렉토리 (app/core/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # app/core/에서 한 번 상위('../')로 app/에 도달한 후 data/로 이동
    file_path = os.path.join(script_dir, '..', 'data', 'documents', '쇼핑몰정보.md')
    loader = TextLoader(file_path,encoding="utf-8")
    return loader.load()
