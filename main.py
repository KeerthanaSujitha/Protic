import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Load the trained model from the pickle file
model = pickle.load(open("trained_model.pkl", 'rb'))

# Text preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

def predict_ticket_type(ticket_subject, ticket_description):
    input_data = preprocess_text(ticket_subject + " " + ticket_description)
    processed_input = [input_data]
    predicted_output = model.predict(processed_input)
    return predicted_output[0]

def main():
    st.title("Ticket Type Prediction")
    
    ticket_subject = st.text_input("Enter Ticket Subject:")
    ticket_description = st.text_area("Enter Ticket Description:")
    
    if st.button("Predict"):
        if ticket_subject and ticket_description:
            predicted_ticket_type = predict_ticket_type(ticket_subject, ticket_description)
            st.success(f"Predicted Ticket Type: {predicted_ticket_type}")
        else:
            st.warning("Please enter both Ticket Subject and Ticket Description.")

if __name__ == '__main__':
    main()
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)