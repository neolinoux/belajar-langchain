from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model = "text-embedding-3-small")

vector_store = Chroma(
  collection_name="kondef_susenas",
  embedding_function=embeddings_model,
  persist_directory=CHROMA_PATH
)

loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200,
  length_function=len,
  is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)

uuids = [str(uuid4()) for _ in range(len(chunks))]

vector_store.add_documents(documents=chunks, uuids=uuids)