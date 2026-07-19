A helpful file explaining how to launch the app locally or push it safely.

Markdown
# 🤖 Beginner-Friendly RAG Chatbot

This is a lightweight Retrieval-Augmented Generation (RAG) chatbot using Streamlit, LangChain, FAISS, and Google Gemini.

## 🚀 How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
Set your Google Gemini API Key:

Windows: set GOOGLE_API_KEY="your_api_key"

Mac/Linux: export GOOGLE_API_KEY="your_api_key"

Launch the web interface:

Bash
streamlit run app.py

---

## ☁️ Step-by-Step Deployment Instructions for Render

1. **Upload code to GitHub:** 
   * Create a new public or private repository on GitHub named `RAG-Chatbot`.
   * Push all 4 files (`app.py`, `requirements.txt`, `data/knowledge.txt`, `README.md`) into your repository root. Keep the empty folder `vector_db/` out or let the code generate it dynamically as handled in the script.

2. **Deploy on Render:**
   * Go to **Render.com** and log in.
   * Click **New +** and select **Web Service**.
   * Connect your GitHub account and select your `RAG-Chatbot` repository.

3. **Configure Settings:**
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

4. **Add Environment Variable:**
   * Scroll down to the **Environment** section.
   * Click **Add Environment Variable**.
   * Set **Key** as `GOOGLE_API_KEY` and paste your actual Google Gemini API key into the **Value** box.
   * Click **Deploy Web Service**! Your live URL will be ready within a few minutes
