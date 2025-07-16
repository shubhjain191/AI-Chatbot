import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "chatbot_db"),
            user=os.getenv("DB_USER", "chatbot_user"),
            password=os.getenv("DB_PASSWORD", "your_password"),
            cursor_factory=RealDictCursor
        )
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                user_input TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                category VARCHAR(100),
                intent VARCHAR(100),
                embedding vector(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create knowledge graph nodes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                type VARCHAR(100) NOT NULL,
                description TEXT,
                embedding vector(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create knowledge graph relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id SERIAL PRIMARY KEY,
                source_entity_id INTEGER REFERENCES entities(id),
                target_entity_id INTEGER REFERENCES entities(id),
                relationship_type VARCHAR(100) NOT NULL,
                weight FLOAT DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create vector indexes for fast similarity search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS conversations_embedding_idx 
            ON conversations USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS entities_embedding_idx 
            ON entities USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100)
        """)
        
        self.conn.commit()
        cursor.close()