import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()


# Function to transform input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Classifier", "About"])

# Home Page
if page == "Home":
    st.title("Welcome to the Email/SMS Spam Classifier")
    st.write("""
    This application helps you classify whether an SMS or email message is spam or not spam.
    Use the navigation sidebar to go to the classifier page and start classifying messages.
    """)
    st.image("Spam.jpeg", caption="Spam Detection")

# Classifier Page
elif page == "Classifier":
    st.title("Email/SMS Spam Classifier")

    input_sms = st.text_area("Enter the message")

    if st.button('Predict'):
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

# About Page
elif page == "About":
    st.title("About")
    st.write("""
    This application was developed to help users identify spam messages using machine learning.
    It leverages Natural Language Processing (NLP) techniques to preprocess the text and a trained model to make predictions.
    """)
    st.write("""
    **Technologies Used:**
    - Python
    - Streamlit
    - NLTK for text preprocessing
    - Scikit-Learn for model training and prediction
    """)
