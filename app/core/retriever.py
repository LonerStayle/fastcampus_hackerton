from typing import Dict, Tuple, List

from langchain_community.retrievers import BM25Retriever 
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.vectorstore import get_vectorstore
from app.core.documents import get_documents  

_dense_retrievers: Dict[Tuple[str, int, float | None], object] = {}
_bm25_retriever: BM25Retriever | None = None

def get_dense_retriever(
    *,
    k: int = 4,
    search_type: str = "similarity",
    score_threshold: float | None = None,
):
    key = (search_type, k, score_threshold)

    if key not in _dense_retrievers:
        vectorstore = get_vectorstore()

        if score_threshold is not None:
            _dense_retrievers[key] = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": k,
                    "score_threshold": score_threshold,
                },
            )
        else:
            _dense_retrievers[key] = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k},
            )

    return _dense_retrievers[key]


def get_bm25_retriever(k: int = 4) -> BM25Retriever:
    global _bm25_retriever

    if _bm25_retriever is None:
        documents: List[Document] = get_documents()
        _bm25_retriever = BM25Retriever.from_documents(documents)

    _bm25_retriever.k = k
    return _bm25_retriever

def rrf_merge(
    docs_list: List[List[Document]],
    *,
    k: int,
    rrf_k: int = 60,
) -> List[Document]:
    scores: Dict[str, dict] = {}

    for docs in docs_list:
        for rank, doc in enumerate(docs):
            key = doc.page_content  # id 없을 때 현실적인 기준
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1 / (rrf_k + rank + 1)

    ranked = sorted(
        scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    return [item["doc"] for item in ranked[:k]]

class RRFEnsembleRetriever(BaseRetriever):
    bm25_k: int
    dense_k: int
    dense_search_type: str
    dense_score_threshold: float | None
    final_k: int

    def _get_relevant_documents(self, query: str) -> List[Document]:
        bm25 = get_bm25_retriever(k=self.bm25_k)
        dense = get_dense_retriever(
            k=self.dense_k,
            search_type=self.dense_search_type,
            score_threshold=self.dense_score_threshold,
        )

        bm25_docs = bm25.invoke(query)
        dense_docs = dense.invoke(query)
        if not dense_docs and not bm25_docs:
            return []

        return rrf_merge(
            [bm25_docs, dense_docs],
            k=self.final_k,
        )


_ensemble_retrievers: Dict[Tuple, RRFEnsembleRetriever] = {}


def get_ensemble_retriever(
    *,
    bm25_k: int = 20,
    dense_k: int = 8,
    dense_search_type: str = "similarity",
    dense_score_threshold = None,
    final_k: int = 8,
):
    key = (
        ("bm25", bm25_k),
        ("dense", dense_search_type, dense_k),
        ("final", final_k),
    )

    if key not in _ensemble_retrievers:
        _ensemble_retrievers[key] = RRFEnsembleRetriever(
            bm25_k=bm25_k,
            dense_k=dense_k,
            dense_search_type=dense_search_type,
            dense_score_threshold=dense_score_threshold,
            final_k=final_k,
        )

    return _ensemble_retrievers[key]
