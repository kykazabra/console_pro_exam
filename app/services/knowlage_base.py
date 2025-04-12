from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings


def get_subpages(url: str) -> List[str]:
    """
    Достает все под страницы с основной
    """

    loader = WebBaseLoader(url)
    soup = loader.scrape()
    links = soup.find_all('a')

    urls = []

    for link in links:
        href = link.get('href')
        if href and href.strip() != '/':
            urls.append(url + href)

    return urls


def load_info(url: str) -> None:
    """
    Заполняет векторное хранилище чанками из указанных URL.
    """

    vector_store = Chroma(
        collection_name="main_collection",
        persist_directory=settings.CHROMA_PATH,
        embedding_function=OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_URL,
            # model=settings.OPENAI_EMBEDDING_MODEL
        )
    )

    print('Качаю основную страницу!')

    docs = WebBaseLoader(url).load()

    print('Качаю под-страницы!')

    for sub_url in get_subpages(url):
        docs += WebBaseLoader(sub_url).load()

    print('Бью на чанки!')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )

    doc_splits = text_splitter.split_documents(docs)

    print('Добавляю в БД!')

    vector_store.reset_collection()
    vector_store.add_documents(doc_splits)

    print('Готово!')





