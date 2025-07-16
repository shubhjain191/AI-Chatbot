import argparse
from src.data_processor import DataProcessor
from src.chatbot import ChatBot
import os

def main():
    parser = argparse.ArgumentParser(description='AI Chatbot with PostgreSQL and pgvector')
    parser.add_argument('--mode', choices=['setup', 'chat', 'process'], 
                       default='chat', help='Mode to run the application')
    parser.add_argument('--data-file', type=str, 
                       help='Path to the dataset file')
    
    args = parser.parse_args()
    
    if args.mode == 'setup':
        print("Setting up database and processing data...")
        processor = DataProcessor()
        
        if args.data_file:
            df = processor.load_dataset(args.data_file)
            processor.process_and_store_data(df)
            print("Data processing completed!")
        else:
            print("Please provide --data-file argument for setup mode")
    
    elif args.mode == 'process':
        print("Processing additional data...")
        processor = DataProcessor()
        if args.data_file:
            df = processor.load_dataset(args.data_file)
            processor.process_and_store_data(df)
            print("Data processing completed!")
        else:
            print("Please provide --data-file argument")
    
    elif args.mode == 'chat':
        print("Starting chatbot interface...")
        chatbot = ChatBot()
        chatbot.run_streamlit_app()

if __name__ == "__main__":
    main()