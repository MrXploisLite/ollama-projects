# Ollama Chatbot

A modern GUI chatbot interface for Ollama with real-time responses and chat management.

## Features
- Modern dark-themed interface
- Real-time chat interaction
- Model selection
- Chat history management
- Progress indication for responses
- Save chat logs to JSON files

## Prerequisites
- Windows 11 
- Ollama ( llama3 )
- Python
- pip

## Setup
1. Ensure Ollama is running on localhost:11434
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Chatbot
```bash
python ollama_chatbot.py
```

## Usage
- Type your message in the input box and press Enter or click Send
- Use the model dropdown to switch between different Ollama models
- Click "Clear Chat" to reset the conversation
- Click "Save Chat" to save the conversation history as JSON

## Notes
- Default model is 'llama3'
- Chat logs are saved in the 'chat_logs' directory
- Model responses are configured with:
  - temperature: 0.7 (creativity)
  - top_p: 0.9 (diversity)
  - top_k: 40 (focus)
