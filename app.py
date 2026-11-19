"""
AI-Powered Spam Email Detection - Streamlit Dashboard
------------------------------------------------------
This file is ONLY a frontend/dashboard. It does NOT train, retrain,
or modify any machine learning model.

It simply LOADS your already-trained files:
    - logistic_model.pkl        (trained Logistic Regression model)
    - tfidf_vectorizer.pkl      (fitted TF-IDF vectorizer)

and uses them to make predictions on messages typed by the user.

Run with:
    streamlit run app.py

Required packages (install once):
    pip install streamlit scikit-learn pandas numpy
"""

import streamlit as st
import pickle
import os
import numpy as np

# ----------------------------------------------------------------------
# 1. PAGE CONFIG  (must be the first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Spam Email Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# 2. CUSTOM CSS (for a professional dashboard look)
# ----------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall page padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.25);
    }
    .header-banner h1 {
        margin-bottom: 0.3rem;
        font-size: 2.2rem;
    }
    .header-banner p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
    }

    /* Section card */
    .section-card {
        background-color: #ffffff;
        border: 1px solid #eef0f4;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        color: #1f2937;
    }

    /* Result cards */
    .result-spam {
        background: linear-gradient(90deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 8px solid #DC2626;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-top: 0.8rem;
    }
    .result-ham {
        background: linear-gradient(90deg, #D1FAE5 0%, #A7F3D0 100%);
        border-left: 8px solid #059669;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-top: 0.8rem;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    /* Example message buttons spacing */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* Metric card tweaks */
    [data-testid="stMetric"] {
        background-color: #f9fafb;
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid #eef0f4;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 3. LOAD THE EXISTING TRAINED MODEL + VECTORIZER  (cached so it loads once)
# ----------------------------------------------------------------------
MODEL_PATH = "logistic_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


@st.cache_resource(show_spinner="Loading trained model and vectorizer...")
def load_model_and_vectorizer():
    """
    Loads the already-trained Logistic Regression model and TF-IDF
    vectorizer from disk. This function does NOT train anything.
    Returns (model, vectorizer, error_message).
    If something goes wrong, error_message will describe it in
    plain English (no raw traceback shown to the user).
    """
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model file not found: '{MODEL_PATH}'. " \
                            f"Please make sure it is in the same folder as app.py."
    if not os.path.exists(VECTORIZER_PATH):
        return None, None, f"Vectorizer file not found: '{VECTORIZER_PATH}'. " \
                            f"Please make sure it is in the same folder as app.py."
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, None
    except Exception:
        # We deliberately hide the raw technical traceback from the user
        # and show a friendly message instead.
        return None, None, ("The model or vectorizer file appears to be "
                             "corrupted or incompatible. Please re-check "
                             "'logistic_model.pkl' and 'tfidf_vectorizer.pkl'.")


model, vectorizer, load_error = load_model_and_vectorizer()

# ----------------------------------------------------------------------
# 4. SESSION STATE (keeps track of the text box + prediction across clicks)
# ----------------------------------------------------------------------
if "message_text" not in st.session_state:
    st.session_state.message_text = ""
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None  # will hold dict with results

# Example messages the user can click to auto-fill the input box
EXAMPLE_SPAM_MESSAGES = [
    "Congratulations! You have WON a $1000 Walmart gift card. Click here to claim now!!!",
    "URGENT: Your account has been suspended. Verify your details immediately to avoid closure.",
    "FREE entry into our $5000 prize draw, just text WIN to 80086 now!",
]
EXAMPLE_HAM_MESSAGES = [
    "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "Don't forget to bring the project report to class tomorrow.",
    "Thanks for helping me with the assignment yesterday, really appreciate it!",
]


def set_message(text):
    """Callback used by example buttons to fill the text area."""
    st.session_state.message_text = text
    st.session_state.prediction_result = None  # reset old result


def clear_all():
    """Callback for the Clear/Reset button."""
    st.session_state.message_text = ""
    st.session_state.prediction_result = None


# ----------------------------------------------------------------------
# 5. HEADER
# ----------------------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <h1>📧 AI-Powered Spam Email Detection</h1>
    <p>A Machine Learning based dashboard that classifies messages as SPAM or HAM (Not Spam) 
    using a trained Logistic Regression model and TF-IDF text features.</p>
</div>
""", unsafe_allow_html=True)

# If the model failed to load, show a clean error and stop the app here.
if load_error:
    st.error(f"⚠️ Could not load the ML model.\n\n**Reason:** {load_error}")
    st.stop()

# ----------------------------------------------------------------------
# 6. SIDEBAR — OVERVIEW / ABOUT
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("📌 Overview")
    st.write(
        "This dashboard is a simple, presentation-friendly interface for a "
        "spam detection system built using classic Machine Learning."
    )
    st.markdown("**Pipeline used:**")
    st.markdown(
        "1. Raw message text\n"
        "2. TF-IDF Vectorization\n"
        "3. Logistic Regression Classifier\n"
        "4. SPAM / HAM prediction + probability"
    )

    st.markdown("---")
    st.header("ℹ️ About the Project")
    st.write(
        "- **Dataset:** SMSSpamCollection (~5,574 messages)\n"
        "- **Algorithm:** Logistic Regression\n"
        "- **Feature Extraction:** TF-IDF\n"
        "- **Accuracy:** ~96.05%\n"
        "- **Purpose:** College PBL demonstration"
    )
    st.caption("Note: This app only builds the front-end. The ML model, "
               "dataset, and training pipeline already existed and were "
               "not changed.")

# ----------------------------------------------------------------------
# 7. MESSAGE ANALYZER SECTION
# ----------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">✍️ Message Analyzer</div>', unsafe_allow_html=True)
st.write("Type or paste a message below, or pick an example from the section further down, then click **Check Message**.")

message_input = st.text_area(
    "Enter the message / email text here:",
    key="message_text",
    height=140,
    placeholder="e.g. Congratulations! You've won a free prize, click here to claim...",
)

col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    check_clicked = st.button("🔍 Check Message", use_container_width=True, type="primary")
with col_btn2:
    st.button("🧹 Clear / Reset", use_container_width=True, on_click=clear_all)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 8. RUN PREDICTION WHEN BUTTON IS CLICKED
# ----------------------------------------------------------------------
if check_clicked:
    text_to_check = st.session_state.message_text.strip()

    if text_to_check == "":
        st.warning("⚠️ Please enter a message before clicking 'Check Message'.")
        st.session_state.prediction_result = None
    else:
        try:
            # Transform the text using the EXISTING fitted TF-IDF vectorizer
            vectorized_text = vectorizer.transform([text_to_check])

            # Predict class (0 = ham, 1 = spam) -- adjust mapping below if needed
            prediction = model.predict(vectorized_text)[0]

            # Predict probability for each class
            probabilities = model.predict_proba(vectorized_text)[0]

            # Most sklearn pipelines trained on this dataset use classes_ = [0, 1]
            # where 1 = spam. We read classes_ directly so we don't hardcode wrongly.
            classes = list(model.classes_)
            spam_label = 1 if 1 in classes else classes[-1]
            spam_index = classes.index(spam_label)
            spam_probability = probabilities[spam_index] * 100
            ham_probability = 100 - spam_probability

            is_spam = (prediction == spam_label)

            # Try to find top influential words (optional, simple, safe)
            top_words = []
            try:
                feature_names = vectorizer.get_feature_names_out()
                coefficients = model.coef_[0]
                # Get indices of words present in this message (non-zero TF-IDF)
                row = vectorized_text.toarray()[0]
                present_indices = np.nonzero(row)[0]
                if len(present_indices) > 0:
                    # Score = tfidf value * model coefficient (contribution toward spam)
                    scored = [(feature_names[i], row[i] * coefficients[i]) for i in present_indices]
                    scored.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_words = scored[:5]
            except Exception:
                top_words = []  # optional feature -- fail silently if anything differs

            st.session_state.prediction_result = {
                "text": text_to_check,
                "is_spam": is_spam,
                "spam_probability": spam_probability,
                "ham_probability": ham_probability,
                "top_words": top_words,
            }
        except Exception:
            st.error("⚠️ Something went wrong while making the prediction. "
                      "Please make sure the model and vectorizer match the expected format.")
            st.session_state.prediction_result = None

# ----------------------------------------------------------------------
# 9. PREDICTION RESULT SECTION
# ----------------------------------------------------------------------
result = st.session_state.prediction_result
if result is not None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

    if result["is_spam"]:
        st.markdown(f"""
        <div class="result-spam">
            <div class="result-title">🚫 SPAM Detected</div>
            <p>This message shows patterns commonly associated with spam.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-ham">
            <div class="result-title">✅ HAM (Not Spam)</div>
            <p>This message looks like a normal, legitimate message.</p>
        </div>
        """, unsafe_allow_html=True)

    # Probability display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Spam Probability", f"{result['spam_probability']:.2f}%")
        st.progress(min(max(result['spam_probability'] / 100, 0.0), 1.0))
    with col2:
        st.metric("Ham Probability", f"{result['ham_probability']:.2f}%")
        st.progress(min(max(result['ham_probability'] / 100, 0.0), 1.0))

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Prediction Explanation ----------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Prediction Explanation</div>', unsafe_allow_html=True)
    st.write(
        "This prediction is based purely on **text patterns and word statistics** "
        "learned by the Logistic Regression model during training on thousands of "
        "labelled messages. It does **not** understand meaning, context, or intent "
        "the way a human would — it recognizes patterns (like certain words, phrases, "
        "or word combinations) that were frequently associated with spam or ham in "
        "the training data."
    )
    if result["top_words"]:
        st.write("**Words in this message that most influenced the prediction:**")
        for word, score in result["top_words"]:
            direction = "→ pushes toward SPAM" if score > 0 else "→ pushes toward HAM"
            st.write(f"- `{word}` {direction}")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 10. EXAMPLE MESSAGES SECTION
# ----------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧪 Example Messages</div>', unsafe_allow_html=True)
st.write("Click any example below to auto-fill the message box above.")

ex_col1, ex_col2 = st.columns(2)
with ex_col1:
    st.markdown("**🚫 Example Spam Messages**")
    for i, msg in enumerate(EXAMPLE_SPAM_MESSAGES):
        st.button(f"📩 {msg[:45]}...", key=f"spam_ex_{i}", on_click=set_message, args=(msg,), use_container_width=True)

with ex_col2:
    st.markdown("**✅ Example Ham Messages**")
    for i, msg in enumerate(EXAMPLE_HAM_MESSAGES):
        st.button(f"📩 {msg[:45]}...", key=f"ham_ex_{i}", on_click=set_message, args=(msg,), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 11. MODEL PERFORMANCE SECTION
# ----------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Model Performance</div>', unsafe_allow_html=True)

perf_col1, perf_col2, perf_col3 = st.columns(3)
with perf_col1:
    st.metric("Model Used", "Logistic Regression")
with perf_col2:
    st.metric("Accuracy", "96.05%")
with perf_col3:
    st.metric("Dataset Size", "~5,574 messages")

st.caption(
    "These performance figures come from the existing training pipeline "
    "(train_model.py / compare_models.py) and are shown here for reference only. "
    "This dashboard does not recompute or alter them."
)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 12. FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit · AI-Powered Spam Email Detection · College PBL Project")
