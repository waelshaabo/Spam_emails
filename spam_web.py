import streamlit as st
import joblib

model=joblib.load('spam_model.joblib')
vectorizer=joblib.load('vectorizer.joblib')

def predict_spam(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction

st.title('Spam Message Classification')
st.write('Enter a message below to check if it is spam or not.')

message = st.text_area('Message',height=100)

if st.button('Check Message'):
    if message:
        result = predict_spam(message)
        if result ==1:
            st.error('This is a spam message!')
        else:
            st.success('This message looks legitimate (Not Spam).')

        st.info('Note: this is a simple model and may not catch all spam messages. Always use your judgment.')
    else:
        st.warning('Please enter a message to check.')