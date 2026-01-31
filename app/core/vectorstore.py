from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

_embeddings = None
_vectorstore = None


def get_vectorstore():
    global _vectorstore, _embeddings

    if _vectorstore is None:
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )

        _vectorstore = Chroma(
            persist_directory="./chroma_md",
            embedding_function=_embeddings,
        )

    return _vectorstore