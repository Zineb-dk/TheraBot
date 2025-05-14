import streamlit as st
from generate import GenerateResponse
import speech_recognition as sr
import time

import streamlit as st
from streamlit_float import *

# initialize float feature/capability
float_init()

#initialize generator
generator = GenerateResponse()

st.markdown("<h1 style='text-align : center;'>TheraBot</h1>", unsafe_allow_html=True)

def main():
    # Initialize chat history and recording state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create footer container and add content
    input_placeholder = st.container()

    with input_placeholder.container():
        col1, col2 = st.columns([11, 1], gap="small")
        with col1:
            user_query = st.chat_input("What is up?", key="user_input")
        with col2:
            recorder = st.button("🎙️")

    # Float the footer container and provide CSS to target it with
    input_placeholder.float("bottom: 0;height:70px;")

    # Accept user input from text or transcription
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response, and add it to the chat history
        response = generator.generate_answer(user_query,chat_history=st.session_state.chat_history)

        # Display the generated response
        with st.chat_message("assistant"):
            st.markdown(response)
        # Update chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    # Handle recorder button functionality
    elif recorder:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.markdown("You can start talking...")
            recognizer.adjust_for_ambient_noise(source, duration=0.2)  
            audio = recognizer.listen(source)
            try:
                user_query = recognizer.recognize_google(audio)
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_query)
                # Generate response, and add it to the chat history
                response = generator.generate_answer(user_query,chat_history=st.session_state.chat_history)

                # Display the generated response
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Update chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except:
                st.markdown("Sorry, I did not get that")

if __name__ == "__main__":
    main()
