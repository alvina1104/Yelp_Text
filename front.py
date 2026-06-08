import streamlit as st
import requests

api_url = "http://127.0.0.1:8000/predict"

st.title("Yelp Review")

user_input = st.text_area("Enter a review")

if st.button("Submit"):
    if user_input.strip():
        response = requests.post(api_url,json={"word": user_input})

        if response.status_code == 200:
            data = response.json()
            st.success(f"Review: {data['prediction']}")
        else:
            st.error("Error connecting to API.")