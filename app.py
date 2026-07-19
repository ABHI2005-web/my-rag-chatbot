import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- STEP 1: SET UP APP CONFIGURATION ---
# Just basic page configuration for our cool Streamlit interface
st.set_page_config(page_title="Friendly RAG Chatbot", layout="centered")
st.title("🤖 My First RAG Chatbot")
st.write("Hey there! Ask me anything based on the document provided inside the data folder.")

# We need an API key for the Gemini LLM. We will fetch it safely from Render environment variables.
# For local testing, you can temporarily paste your key here like: GOOGLE_API_KEY = "AIzaSy..."
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.warning("⚠️ Uh oh! Google API Key is missing. Please set GOOGLE_API_KEY in your Render environment variables.")
    st.stop()

# --- STEP 2: DEFINE PATHS & INITIALIZE FIELDS ---
# Let's set up the directories where our documents are and where our vector database will be saved.
DATA_PATH = "data/knowledge.txt"
DB_FAISS_PATH = "vector_db/db_faiss"

# --- STEP 3: PREPROCESSING & VECTOR STORE CREATION FUNCTION ---
@st.cache_resource
def initialize_rag_system():
    """
    This function loads our text file, chops it into tiny chunks, 
    turns those chunks into math vectors using embeddings, and saves it locally.
    """
    # 1. Check if the file even exists first!
    if not os.path.exists(DATA_PATH):
        # If no file exists, let's create a quick dummy one so the application doesn't crash
        os.makedirs("data", exist_ok=True)
        with open(DATA_PATH, "w") as f:
            f.write("Hello! This is a backup knowledge base. The RAG system is working fine!")
    
    # 2. Load the document text using LangChain's TextLoader
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
    
    # 3. Chop the text into smaller parts (chunking) so the model can read it easily
    # We use RecursiveCharacterTextSplitter as per our reference guide instructions!
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # 4. Grab the embedding model recommended in our reference guide: all-MiniLM-L6-v2
    # This turns plain text sentences into lists of numbers that represent meanings.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 5. Build the FAISS Vector Database using our chopped text documents and embeddings
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # 6. Save the database locally so we don't have to rebuild it every single time the user clicks something
    vector_store.save_local(DB_FAISS_PATH)
    
    return vector_store

# Let's actually trigger the system setup function now!
try:
    vector_store = initialize_rag_system()
    # Let's turn our vector store into a retriever that searches for similar text chunks matching user questions
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
except Exception as e:
    st.error(f"Something went wrong while setting up the knowledge base: {e}")
    st.stop()

# --- STEP 4: SET UP THE GENERATIVE LLM & PROMPT CHAIN ---
# We connect to Google Gemini (specifically gemini-1.5-flash as it's lightning fast and free)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)

# Design a clear, natural systemic instruction telling the AI how to behave.
system_prompt = (
    "You are a helpful, polite assistant. Use the following pieces of retrieved context to answer "
    "the user's question. If you don't know the answer, just say you don't know clearly, do not try to make up a story.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the chains: first stuff all document contexts into the prompt template, then plug it into the retriever chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- STEP 5: STREAMLIT CHAT UI ---
# Initialize simple session memory state for keeping chat history on the screen
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages on the screen if the user is having a long chat session
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Wait for the user to type a question into the input chat box
if user_query := st.chat_input("Type your question here..."):
    
    # Immediately display what the user just typed
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    # Show a friendly loading spinner while the RAG chain does the brain work
    with st.chat_message("assistant"):
        with st.spinner("Let me look through my notes real quick..."):
            try:
                # Query the system! It searches the FAISS DB and forwards relevant texts to Gemini
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                
                # Render the final text answer gracefully on the screen
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Oops! I hit an error trying to process that: {e}")
