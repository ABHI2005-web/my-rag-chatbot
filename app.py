import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- STEP 1: SET UP APP CONFIGURATION ---
st.set_page_config(page_title="Friendly RAG Chatbot", layout="centered")
st.title("🤖 My First RAG Chatbot")
st.write("Hey there! Ask me anything based on the document provided inside the data folder.")

# Fetch the API key safely from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.warning("⚠️ Uh oh! Google API Key is missing. Please set GOOGLE_API_KEY in your Render environment variables.")
    st.stop()

# --- STEP 2: DEFINE PATHS & INITIALIZE FIELDS ---
DATA_PATH = "data/knowledge.txt"
DB_FAISS_PATH = "vector_db/db_faiss"

# --- STEP 3: PREPROCESSING & VECTOR STORE CREATION FUNCTION ---
@st.cache_resource
def initialize_rag_system():
    """
    Loads text, splits into segments, and computes embeddings using Google GenAI cloud service.
    """
    # 1. Check if the directory and file exist
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        with open(DATA_PATH, "w") as f:
            f.write("Hello! This is a backup knowledge base. The RAG system is working fine!")
    
    # 2. Load the document text
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
    
    # 3. Chop the text into pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # 4. Corrected Model API mapping for cloud-based embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY) 
    
    # 5. Build the FAISS database index
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # 6. Save locally inside the container
    vector_store.save_local(DB_FAISS_PATH)
    
    return vector_store

# Trigger the database setup
try:
    vector_store = initialize_rag_system()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
except Exception as e:
    st.error(f"Something went wrong while setting up the knowledge base: {e}")
    st.stop()

# --- STEP 4: SET UP THE GENERATIVE LLM & PROMPT CHAIN ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)

system_prompt = (
    "You are a helpful, polite assistant. Use the following pieces of retrieved context to answer "
    "the user's question. If you don't know the answer, just say you don't know clearly, do not try to make up a story.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- STEP 5: STREAMLIT CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_query := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Let me look through my notes real quick..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Oops! I hit an error trying to process that: {e}")
    
    
