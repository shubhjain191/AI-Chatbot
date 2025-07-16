import pandas as pd
import json
from typing import Dict, List, Tuple
import re
from src.database import DatabaseConnection
from src.embeddings import EmbeddingService

class DataProcessor:
    def __init__(self):
        self.db = DatabaseConnection()
        self.embedding_service = EmbeddingService()

    def load_customer_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load the customer support dataset from a CSV file.
        """
        customer_dataset = pd.read_csv(file_path)
        return customer_dataset

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text by removing extra whitespace and unwanted characters.
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        cleaned_text = text.strip()
        return cleaned_text

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract simple entities from text using regex patterns.
        """
        entities = []
        patterns = {
            'PRODUCT': r'\b(account|subscription|service|plan|package)\b',
            'ISSUE': r'\b(problem|issue|error|bug|complaint)\b',
            'ACTION': r'\b(cancel|refund|upgrade|downgrade|reset)\b',
            'EMOTION': r'\b(frustrated|happy|angry|satisfied|disappointed)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text.lower())
            for match in matches:
                entities.append({
                    'name': match,
                    'type': entity_type,
                    'description': f'{entity_type.lower()} mentioned in customer interaction'
                })
        
        return entities

    def process_and_store(self, df: pd.DataFrame):
        """
        Process the dataset and store conversations and entities in the database.
        """
        db_operator = self.db.conn.cursor()
        for _, row in df.iterrows():
            user_input = self.clean_text(row.get('user_input', ''))
            bot_response = self.clean_text(row.get('bot_response', ''))
            category = row.get('category', 'general')
            intent = row.get('intent', 'unknown')
            
            embedding = self.embedding_service.encode_text(user_input)[0]

            # Store conversation in the database
            db_operator.execute(
                """
                INSERT INTO conversations (user_input, bot_response, category, intent, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_input, bot_response, category, intent, embedding.tolist())
            )

            # Extract and store entities
            entities = self.extract_entities(user_input + " " + bot_response)
            for entity in entities:
                entity_embedding = self.embedding_service.encode_text(entity['name'])[0]
                db_operator.execute(
                    """
                    INSERT INTO entities (name, type, description, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name) DO NOTHING
                    """,
                    (entity['name'], entity['type'], entity['description'], entity_embedding.tolist())
                )
        self.db.conn.commit()
        db_operator.close()