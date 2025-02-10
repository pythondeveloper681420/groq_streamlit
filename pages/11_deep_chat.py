import streamlit as st
from groq import Groq
import time

# Page configuration
st.set_page_config(page_title="AI Chatbot with Groq", page_icon="💭")

# Initialize session state for conversation history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

def initialize_groq():
    """Initialize Groq API with error handling"""
    try:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
        return client
    except Exception as e:
        st.error("Error initializing Groq API. Please check your API key configuration.")
        return None

def get_chatbot_response(client, user_input, retry_attempts=3):
    """Get response from Groq with retry logic and error handling"""
    for attempt in range(retry_attempts):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    *st.session_state.messages,
                    {"role": "user", "content": user_input}
                ],
                model="mixtral-8x7b-32768",  # You can also use "llama2-70b-4096"
                temperature=0.7,
                max_tokens=1000
            )
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            if "rate limit" in str(e).lower():
                st.warning("Rate limit reached. Please check your Groq API quota and billing details.")
                return None
                
            if attempt < retry_attempts - 1:  # Don't sleep on the last attempt
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            st.error(f"An error occurred: {str(e)}")
            return None
    
    return None

def main():
    st.title("💭 AI Chatbot using Groq")
    st.markdown("""
    Welcome to the AI Chatbot! This version uses Groq's API for faster response times.
    """)
    
    # Initialize Groq
    client = initialize_groq()
    if not client:
        return
    
    # Display conversation history
    for message in st.session_state.messages:
        role = "🤖 Bot" if message["role"] == "assistant" else "👤 You"
        st.write(f"{role}: {message['content']}")
    
    # User input
    user_input = st.text_input("Type your message:", key="user_input")
    
    # Send button
    if st.button("Send", key="send"):
        if not user_input.strip():
            st.warning("Please enter a message.")
            return
            
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.spinner("Thinking..."):
            bot_response = get_chatbot_response(client, user_input)
            
        if bot_response:
            # Add bot response to history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            # Force refresh to show new messages
            st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()