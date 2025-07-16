import streamlit as st
from src.rag_system import RAGSystem
from src.knowledge_graph import KnowledgeGraphBuilder
import os

class ChatBot:
    def __init__(self):
        self.rag_system = RAGSystem()
        self.kg_builder = KnowledgeGraphBuilder()
    
    def run_streamlit_app(self):
        st.set_page_config(
            page_title="AI Customer Support Chatbot",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 AI Customer Support Chatbot")
        st.write("Ask me anything about our services!")
        
        # Sidebar for configuration
        with st.sidebar:
            st.header("Configuration")
            if st.button("Rebuild Knowledge Graph"):
                with st.spinner("Building knowledge graph..."):
                    self.kg_builder.build_entity_relationships()
                    st.success("Knowledge graph updated!")
            
            if st.button("Visualize Knowledge Graph"):
                with st.spinner("Generating visualization..."):
                    self.kg_builder.visualize_graph()
                    st.success("Graph saved as knowledge_graph.png")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your question here..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.rag_system.generate_response(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.run_streamlit_app()