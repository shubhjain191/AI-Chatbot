import numpy as np
from typing import List, Dict
from src.database import DatabaseConnection
from src.embeddings import EmbeddingService
import os
import google.generativeai as genai

class RAGSystem:
    def __init__(self):
        self.db = DatabaseConnection()
        self.embedding_service = EmbeddingService()

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    def get_similar_conversations(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Find top_k conversations similar to the user's question by comparing embeddings.
        """
        question_embedding = self.embedding_service.encode_text(question)[0]
        db_operator = self.db.conn.cursor()
        db_operator.execute(
            """
            SELECT user_input, bot_response, category, intent,
                   embedding <-> %s AS distance
            FROM conversations
            ORDER BY embedding <-> %s
            LIMIT %s
            """,
            (question_embedding, question_embedding, top_k)
        )
        results = db_operator.fetchall()
        db_operator.close()
        return [dict(row) for row in results]
    
    def entities_related_to_question(self, question: str, top_k: int = 3) -> List[Dict]:
        """
        Find top_k entities related to the user's question using embedding similarity.
        """
        question_embedding = self.embedding_service.encode_text(question)[0]
        db_operator = self.db.conn.cursor()
        db_operator.execute(
            """
            SELECT name, type, description,
                   embedding <-> %s AS distance
            FROM entities
            ORDER BY embedding <-> %s
            LIMIT %s
            """,
            (question_embedding, question_embedding, top_k)
        )
        results = db_operator.fetchall()
        db_operator.close()
        return [dict(row) for row in results]
    
    def generate_response(self, question: str) -> str:
        """
        Use retrieved context and Gemini AI to generate a helpful answer for the user query.
        """
        similar_conversations = self.get_similar_conversations(question)
        related_entities = self.get_related_entities(question)
        context_text = "Previous similar conversations:\n"
        for convo in similar_conversations:
            context_text += f"Q: {convo['user_input']}\nA: {convo['bot_response']}\n\n"
        context_text += "Related entities:\n"
        for entity in related_entities:
            context_text += f"- {entity['name']} ({entity['type']}): {entity['description']}\n"
        

        prompt_text = (
            "You are a helpful customer support chatbot. "
            "Use the provided context to answer the user's question as accurately and helpfully as possible."
            "Context:"
            f"{context_text}"
            "User Question:"
            f"{question}"
            "Based on the above context, please provide a detailed and helpful response to the user's question."
        )

        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt_text)
            return response.text
        except Exception as e:
            return f"Sorry, I am having trouble generating a response right now! Error: {str(e)}"
