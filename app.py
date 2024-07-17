import fitz
import re
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed(query, tokenizer, model_encoder):
    encoded_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model_encoder(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def top_sentences(query, embeddings, sentences, tokenizer, model_encoder):
    query_embedding = embed([query], tokenizer, model_encoder)
    similarities = F.cosine_similarity(query_embedding, embeddings).flatten()
    cutoff_score = 0.2
    top_idx = similarities.argsort(descending=True)[:8]
    return [sentences[i] for i in top_idx]

# Streamlit app
st.title("PDF-based Q&A System")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    text = extract_text_from_pdf(pdf_path)

    st.write("Extracted Text Preview:")
    st.write(text[:100])

    sentences = re.split(r'[.!?;]', text)
    sentences = [preprocess_text(sentence) for sentence in sentences]

    # Load models
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model_encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    embeddings = embed(sentences, tokenizer, model_encoder)
    
    query = st.text_input("Enter your query:")
    if query:
        context = top_sentences(query, embeddings, sentences, tokenizer, model_encoder)

        gemini_api_key = 'YOUR_API_KEY'
        genai.configure(api_key=gemini_api_key)

        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model_genai = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
            system_instruction="You will be provided with context from a PDF document. Analyze the context and respond to queries based on the information given.",
        )

        chat_session = model_genai.start_chat(
            history=[]
        )

        chat_session = model_genai.start_chat(
            history=[]
        )

        context = top_sentences(query, embeddings, sentences)
        message = {
            "role" : "user",
            "parts" : [
                f"Context: {context}",
                f"query : {query}"
            ]
        }

        response = chat_session.send_message(message)

        if response:
            answer = response["choices"][0]["message"]["content"]
            st.write("Answer:")
            st.write(answer)
