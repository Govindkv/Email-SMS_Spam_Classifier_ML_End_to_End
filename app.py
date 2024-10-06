import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Initialize PorterStemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize text
    y = []
    for i in text:
        if i.isalnum():  # Keep alphanumeric characters
            y.append(i)
    
    text = y[:]  # Copy list
    y.clear()
    
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)  # Remove stopwords and punctuation

    text = y[:]  # Copy list
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Perform stemming

    return " ".join(y)  # Return the processed text

# Load vectorizer and model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input text area
input_sms = st.text_area("Enter the message")

# Predict button
if st.button("Predict"):
    # 1. Preprocess the input
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the input
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict the result
    result = model.predict(vector_input)[0]
    
    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
