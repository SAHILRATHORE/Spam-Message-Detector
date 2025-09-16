import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# --- Page Config ---
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }
    .stTextArea textarea {
        font-size: 16px !important;
        border-radius: 10px !important;
        border: 2px solid #4CAF50 !important;
        padding: 10px !important;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        padding: 10px 20px !important;
        transition: 0.3s !important;
    }
    .stButton>button:hover {
        background-color: #45a049 !important;
        transform: scale(1.05);
    }
    .result-box {
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
    }
    .spam {
        background-color: #ffdddd;
        color: #d8000c;
        border: 1px solid #d8000c;
    }
    .ham {
        background-color: #ddffdd;
        color: #4f8a10;
        border: 1px solid #4f8a10;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("📩 Spam Message Detector")
st.subheader("Detect spam messages instantly using Machine Learning ⚡")

st.write(
    "This app uses a **Multinomial Naive Bayes model** trained on SMS messages. "
    "Type a message below and click **Predict** to see if it's Spam or Ham 🚀."
)

# --- User Input ---
message = st.text_area("✍️ Enter your message here:", "", height=150)

# --- Prediction ---
if st.button("🔍 Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        transformed = vectorizer.transform([message])
        prediction = model.predict(transformed)[0]
        proba = model.predict_proba(transformed)[0]
        confidence = max(proba) * 100

        if prediction == "spam":
            st.markdown(
                f"<div class='result-box spam'>🚨 This message is <b>SPAM</b> ❌<br>🔎 Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True,
            )
            st.markdown("💡 **Tip:** Be careful with suspicious links, offers, and unknown senders.")
        else:
            st.markdown(
                f"<div class='result-box ham'>✅ This message is <b>HAM (Not Spam)</b> 🎉<br>🔎 Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True,
            )
            st.markdown("👍 **Safe:** This looks like a normal message.")

# --- Sidebar Info ---
st.sidebar.header("📊 About")
st.sidebar.write("""
- **Model**: Multinomial Naive Bayes  
- **Feature Extraction**: TF-IDF Vectorizer  
- **Balanced with**: Random Oversampling  
""")

# --- Portfolio Button in Sidebar ---
st.sidebar.markdown(
    """
    <a href="https://sahilportfolio605.netlify.app" target="_blank">
        <button style="
            background-color:#4CAF50;
            color:white;
            padding:10px 20px;
            border:none;
            border-radius:8px;
            font-size:16px;
            cursor:pointer;">
            🌐 Visit My Portfolio
        </button>
    </a>
    """,
    unsafe_allow_html=True
)
