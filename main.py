import argparse
import sys
import os
from src.data_processor import DataProcessor
from src.chatbot import ChatBot


def parse_arguments():
    """
    Parse command-line arguments for the chatbot application.
    """
    parser = argparse.ArgumentParser(
        description="AI Chatbot with PostgreSQL and pgvector integration"
    )
    parser.add_argument(
        "--mode",
        choices=["setup", "chat", "process"],
        default="chat",
        help="Mode to run the application: setup, chat, or process. Default is chat."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to the dataset file (required for setup and process modes)."
    )
    args = parser.parse_args()
    return args



def main():
    args = parse_arguments()

    # Setup mode: initialize database and process data
    if args.mode == "setup":
        print("[Info] Setting up database and processing data...")
        processor = DataProcessor()
        if not args.data_file:
            print("[Error] Please provide --data-file argument for setup mode.")
            sys.exit(1)
        if not os.path.exists(args.data_file):
            print(f"[Error] Data file '{args.data_file}' does not exist.")
            sys.exit(1)
        df = processor.load_dataset(args.data_file)
        processor.process_and_store_data(df)
        print("[Success] Data processing completed!")

    # Process mode: process additional data
    elif args.mode == "process":
        print("[Info] Processing additional data...")
        processor = DataProcessor()
        if not args.data_file:
            print("[Error] Please provide --data-file argument for process mode.")
            sys.exit(1)
        if not os.path.exists(args.data_file):
            print(f"[Error] Data file '{args.data_file}' does not exist.")
            sys.exit(1)
        df = processor.load_dataset(args.data_file)
        processor.process_and_store_data(df)
        print("[Success] Data processing completed!")

    # Chat mode: start the chatbot interface
    elif args.mode == "chat":
        print("[Info] Starting chatbot interface...")
        chatbot = ChatBot()
        chatbot.run_streamlit_app()
    else:
        print(f"[Error] Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()