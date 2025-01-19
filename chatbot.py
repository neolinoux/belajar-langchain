from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model = "text-embedding-3-small")

llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

vector_store = Chroma(
  collection_name="kondef_susenas",
  embedding_function=embeddings_model,
  persist_directory=CHROMA_PATH
)

num_results = 3
retriever = vector_store.as_retriever(seach_kwargs={'k':num_results})

def stream_response(message, history):
  # print(f"Input : {message}. History : {history}\n")
  
  docs = retriever.invoke(message)
  
  knowledge = ""
  
  for doc in docs:
    knowledge += doc.page_content+"\n\n"
    
  if message is not None:
    partial_message = ""
    
    rag_prompt = f"""Ini adalah pembantu virtual yang membantu Anda menemukan informasi mengenai kondef susenas. 
    
    Pertanyaan : {message}
    
    Histori Percakapan : {history}
    
    Informasi yang didapat : {knowledge}
    
    """
    
    for response in llm.stream(rag_prompt):
      partial_message += response.content
      yield partial_message
        
chatbot = gr.ChatInterface(
  stream_response,
  textbox=gr.Textbox(
    placeholder="Tulis pertanyaan Anda di sini...",
    container=False,
    autoscroll=True,
    scale=7  
  )
)

chatbot.launch()