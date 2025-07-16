import numpy as np
from typing import List, Dict, Tuple
from src.database import DatabaseManager
from src.embeddings import EmbeddingService
import os
import google.generativeai as genai

class RAGSystem:
    def __init__(self):
        self.db = DatabaseManager()
        self.embedding_service = EmbeddingService()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    def retrieve_similar_conversations(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve similar conversations from database"""
        query_embedding = self.embedding_service.encode_text(query)[0]
        
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT user_input, bot_response, category, intent,
                   embedding <-> %s as distance
            FROM conversations
            ORDER BY embedding <-> %s
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))  # Pass raw NumPy array
        
        results = cursor.fetchall()
        cursor.close()
        
        return [dict(row) for row in results]
    
    def retrieve_related_entities(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve related entities from knowledge graph"""
        query_embedding = self.embedding_service.encode_text(query)[0]
        
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT name, type, description,
                   embedding <-> %s as distance
            FROM entities
            ORDER BY embedding <-> %s
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))  # Pass raw NumPy array
        
        results = cursor.fetchall()
        cursor.close()
        
        return [dict(row) for row in results]
    
    def generate_response(self, query: str) -> str:
        """Generate response using RAG"""
        # Retrieve relevant information
        similar_conversations = self.retrieve_similar_conversations(query)
        related_entities = self.retrieve_related_entities(query)
        
        # Build context
        context = "Previous similar conversations:\n"
        for conv in similar_conversations:
            context += f"Q: {conv['user_input']}\nA: {conv['bot_response']}\n\n"
        
        context += "Related entities:\n"
        for entity in related_entities:
            context += f"- {entity['name']} ({entity['type']}): {entity['description']}\n"
        
        # Generate response using Gemini
        prompt = f"""
        You are a helpful customer support chatbot. Use the following context to answer the user's question.
        
        Context:
        {context}
        
        User Question: {query}
        
        Please provide a helpful, accurate response based on the context above.
        """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"