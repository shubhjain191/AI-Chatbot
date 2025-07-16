import streamlit as st
from src.rag_system import RAGSystem
import os

class ChatBot:
    def __init__(self):
        self.rag_system = RAGSystem()
    
    #Streamlit Page Configuration
    def run_streamlit_app(self):
        st.set_page_config(
            page_title="AI Customer Support Chatbot",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 AI Customer Support Chatbot")
        st.write("Ask me anything about our services!")
        
        #Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat Input
        if prompt := st.chat_input("Type your question here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.rag_system.generate_response(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.run_streamlit_app()