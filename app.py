import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Downloading NLTK data
nltk.download('punkt')
nltk.download('stopwords')
# Initializing PorterStemmer
ps = PorterStemmer()

# Text preprocessing function
def text_process(text):
    text = text.lower()  # Lowercasing
    text = nltk.word_tokenize(text)  # Tokenizing into words
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)  # Append only if alphanumeric
    
    text = y[:]  # Cloning
    y.clear()
    
    for i in text:  # Removing stop words and punctuations
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:  # Stemming
        y.append(ps.stem(i))
    
    return " ".join(y)

# Loading the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app
st.set_page_config(page_title="Spam SMS Detection", page_icon="📱")

st.title("📱 Spam SMS Detection")
st.subheader("Detect spam messages using machine learning")

# Explanation section
with st.expander("ℹ️ About this app"):
    st.write("""
        This application uses a machine learning model to classify SMS messages as spam or not spam.
        The model has been trained using natural language processing (NLP) techniques.
        Enter an SMS message below to check if it's spam or not.
    """)

# Text input
input_sms = st.text_area("Enter your SMS message here:")

# Predict button with spinner
if st.button('Predict'):
    if input_sms.strip() != "":
        with st.spinner('Analyzing the message...'):
            # Preprocess the input message
            processed_sms = text_process(input_sms)
            # Vectorize the processed message
            vector_input = tfidf.transform([processed_sms])
            # Predict using the model
            result = model.predict(vector_input)[0]
            
            # Display the result with styling
            if result == 1:
                st.markdown("<h2 style='color: red;'>🚫 Spam</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color: green;'>✅ Not Spam</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please enter an SMS message to classify.")

# Clear button to reset the input field
if st.button('Clear'):
    st.experimental_rerun()

# Styling the app
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: white; 
        color: black; 
        border: 2px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
