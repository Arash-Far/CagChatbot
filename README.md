# Carton Caps AI Assistant

An intelligent AI chatbot built with LangGraph and FastAPI that serves as a virtual assistant for the Carton Caps app. The chatbot helps users with product recommendations and referral program assistance, with a focus on benefiting schools through user engagement.

## Features

- Intelligent AI assistant (Capper) powered by GPT-4.1 Mini
- Real-time streaming chat responses
- Conversation history persistence
- User authentication and session management
- Knowledge base integration (FAQs and Referral Rules)
- Product catalog integration
- Goal-oriented conversation flow

## Architecture

The application consists of:

- FastAPI backend server with:
  - LangGraph for conversation orchestration
  - SQLite database for user and conversation management
  - Streaming response support
  - Knowledge base integration
- Streamlit frontend with:
  - Real-time chat interface
  - User authentication
  - Conversation history management

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Arash-Far/CagChatbot.git
cd CagChatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

## 💻 Usage

1. Start the FastAPI backend server:
```bash
uvicorn server:app --reload --port 8000
```

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run client.py
```

3. Open your browser and navigate to `http://localhost:8501`
4. Register with your email to start chatting with Capper

## ⚙️ Configuration

The chatbot's knowledge base consists of:
- `knowledge/FAQs.txt`: Frequently asked questions and answers
- `knowledge/Referral_Rules.txt`: Referral program rules and guidelines
- Product catalog from the database

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 