# streamlit_app.py

import streamlit as st
import requests
import feedparser
import PyPDF2
import io
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from textblob import TextBlob
from datetime import datetime

# Set seed for langdetect
DetectorFactory.seed = 0

# ---------------------------
# Load models with caching
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    summarizer_model = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    sentiment_analyzer = pipeline("sentiment-analysis",
                                  model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

    return summarizer, sentiment_analyzer

# ---------------------------
# Utility functions
# ---------------------------
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = " ".join(soup.get_text().split())
        return text
    except:
        return None

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except:
        return None

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def summarize_text(text, summarizer):
    try:
        max_len = min(512, len(text) // 4)
        min_len = max(30, max_len // 4)

        if len(text.split()) < 50:
            return text

        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary[0]['summary_text']
    except:
        return "Summary unavailable"

def analyze_sentiment(text, sentiment_analyzer):
    try:
        if len(text.split()) < 10:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.1: return "positive", polarity
            elif polarity < -0.1: return "negative", polarity
            else: return "neutral", polarity

        result = sentiment_analyzer(text[:512])[0]
        label = result['label']
        score = result['score']
        if "positive" in label.lower():
            return "positive", score
        elif "negative" in label.lower():
            return "negative", score
        else:
            return "neutral", score
    except:
        return "unknown", 0

def fetch_rss_feed(feed_url):
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries:
        title = entry.get("title", "No title")
        link = entry.get("link", "")
        summary = entry.get("summary", "")
        preview = BeautifulSoup(summary, "html.parser").get_text()[:200] + "..."
        articles.append({"title": title, "url": link, "preview": preview})
    return articles

def process_article(text, title, source, url, summarizer, sentiment_analyzer):
    language = detect_language(text)
    summary = summarize_text(text, summarizer)
    sentiment, score = analyze_sentiment(text, sentiment_analyzer)

    article = {
        "title": title,
        "source": source,
        "url": url,
        "language": language,
        "summary": summary,
        "sentiment": sentiment,
        "sentiment_score": score,
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_text_preview": text[:500] + "..." if len(text) > 500 else text
    }
    return article

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Multilingual News NLP", layout="wide")
st.title("🌍 Multilingual News Summarization & Sentiment Analysis")

st.markdown("""
This application can:
- Detect language of articles  
- Generate summaries  
- Perform sentiment analysis  

Supports **English, Arabic, French, Spanish, German, Russian, Chinese, and more.**
""")

# Step 1: Load models
st.header("Step 1: Load Models")
with st.spinner("Loading models... This may take a few minutes the first time."):
    summarizer, sentiment_analyzer = load_models()
st.success("✅ Models loaded successfully!")

# Step 2: Input Method
st.header("Step 2: Select Input Method")
method = st.radio("Choose Input Method", ["RSS Feed", "Direct Text", "URL", "File Upload"])

if method == "RSS Feed":
    rss_url = st.text_input("RSS Feed URL", "https://techcrunch.com/feed/")
    if st.button("Fetch & Process RSS Articles"):
        articles = fetch_rss_feed(rss_url)
        if articles:
            for article in articles[:3]:
                text = extract_text_from_url(article['url'])
                if text:
                    result = process_article(text, article['title'], "RSS Feed", article['url'],
                                             summarizer, sentiment_analyzer)
                    st.subheader(result["title"])
                    st.write(f"**Source:** {result['source']}")
                    st.write(f"**Language:** {result['language']}")
                    st.write(f"**Sentiment:** {result['sentiment']} (Score: {result['sentiment_score']:.3f})")
                    st.write(f"**Processed at:** {result['processed_at']}")
                    st.markdown("**Summary:**")
                    st.info(result["summary"])
                    with st.expander("Original Text Preview"):
                        st.write(result["original_text_preview"])
        else:
            st.error("No articles found in RSS feed")

elif method == "Direct Text":
    text_input = st.text_area("Paste article text here", height=200)
    if st.button("Process Text"):
        if text_input.strip():
            result = process_article(text_input, "Direct Input", "Text Input", "",
                                     summarizer, sentiment_analyzer)
            st.json(result)
        else:
            st.warning("Please enter some text")

elif method == "URL":
    url_input = st.text_input("Enter article URL")
    if st.button("Process URL"):
        if url_input:
            text = extract_text_from_url(url_input)
            if text:
                result = process_article(text, "URL Input", url_input, url_input,
                                         summarizer, sentiment_analyzer)
                st.json(result)
            else:
                st.error("Could not extract text from URL")
        else:
            st.warning("Please enter a URL")

elif method == "File Upload":
    uploaded_file = st.file_uploader("Upload a file (.txt or .pdf)", type=["txt", "pdf"])
    if uploaded_file and st.button("Process File"):
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            title = uploaded_file.name
        elif uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file.read())
            title = uploaded_file.name
        else:
            text = None

        if text:
            result = process_article(text, title, "File Upload", "",
                                     summarizer, sentiment_analyzer)
            st.json(result)
        else:
            st.error("Could not extract text from file")

# Footer
st.markdown("---")
st.markdown("⚡ Built with **Streamlit + HuggingFace Transformers**")

