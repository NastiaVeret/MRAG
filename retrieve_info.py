import pickle
from pathlib import Path
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import streamlit as st
import os
from my_metrics import compute_context_relevance, compute_groundedness, evaluate_answer
api_key = os.getenv("API_KEY")
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

text_index = faiss.read_index("text.index")
with open("text_data.pkl", "rb") as f:
    news_items = pickle.load(f)

image_folder = Path("")
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def ask_gemini(query, context_text):
    prompt = f"""You are a helpful assistant. Answer the question strictly based on the provided context.
Do not invent or include any information that is not present in the context. If the answer cannot be found in the context, reply with: "I don't know based on the given context."

Context:
{context_text}

Question:
{query}
"""
    response = gemini.generate_content(
        prompt,
        generation_config={"temperature": 0.0}
    )
    return response.text.strip()

def search(query, top_k=3):
    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = text_index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        item = news_items[idx]
        image_path = image_folder / item.get("image_file", "")
        results.append((item, image_path))
    return results

st.set_page_config(page_title="RAG System Demo", layout="wide")
st.title(" RAG System Demo")

user_query = st.text_input("Enter your question:")
if user_query:
    with st.spinner("Searching and generating response..."):
        results = search(user_query, top_k=3)
        if not results:
            st.warning("No relevant articles found.")
        else:
            item, image_path = results[0]
            full_context = f"{item['title']}\n\n{item['full_content']}"
            answer = ask_gemini(user_query, full_context)

            st.subheader(" Answer")
            st.write(answer)

            st.subheader(" Retrieved Article")
            st.markdown(f"**Title:** {item['title']}")
            st.markdown(f"**Chunk (used for retrieval):** {item.get('chunk', '[no chunk]')}")
            st.markdown("**Full Content:**")
            st.text(item['full_content'])
            if image_path.exists():
                st.image(str(image_path), caption="Relevant image", use_container_width=True)
            else:
                st.info("Image not found.")

            context_relevance = compute_context_relevance(user_query, [item.get("chunk", "")])
            groundedness = compute_groundedness(answer, [item.get("chunk", "")])
            bert_f1 = evaluate_answer(answer, item.get("chunk", ""))

            st.subheader(" Evaluation Metrics")
            st.markdown(f"- **Context Relevance:** `{context_relevance:.3f}`")
            st.markdown(f"- **Groundedness:** `{groundedness:.3f}`")
            st.markdown(f"- **Answer Relevance (BERT F1):** `{bert_f1:.4f}`")
