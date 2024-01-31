import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  # Import joblib directly


# Load the pre-trained model and vectorizer
model = joblib.load('spams_classifier.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to process and classify messages
def classify_messages(df):
    # Extract features
    X_new = df['Text']
    
    # Transform text data using the pre-trained TfidfVectorizer
    X_new_tfidf = tfidf_vectorizer.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_new_tfidf)
    
    # Add prediction column to the DataFrame
    df['Prediction'] = predictions
    
    # Separate spam and non-spam messages
    spam_messages = df[df['Prediction'] == 1]
    non_spam_messages = df[df['Prediction'] == 0]
    
    # Save messages to CSV files
    spam_messages.to_csv('spam.csv', index=False, header=True)
    non_spam_messages.to_csv('non_spam.csv', index=False, header=True)

# Streamlit UI
st.title('Spam Classifier Application')
st.subheader('Task 4 Flazetech')

# File upload
uploaded_file = st.file_uploader('Upload a CSV file with columns Text and Length', type=['csv'])

if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the loaded data
    st.subheader('Uploaded Data:')
    st.write(df.drop(columns=['Label']))
    
    # Process and classify messages
    classify_messages(df)
    
    # Display success message
    st.success('Classification completed successfully!')
    st.info('Check the generated files: spam.csv and non_spam.csv')
