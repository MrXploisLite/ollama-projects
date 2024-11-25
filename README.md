# Ollama Chat GUI

A modern, feature-rich GUI chatbot interface for Ollama with real-time streaming responses and advanced features.

## Features

### Core Features
- Real-time streaming responses
- Multiple model support
- Modern dark theme interface
- Chat history management
- Automatic reconnection

### Advanced Features
- Markdown rendering with syntax highlighting
- Conversation style presets (Balanced, Creative, Precise)
- Custom temperature and parameter controls
- Context visualization and management
- Token estimation and limits

### User Experience
- Keyboard shortcuts
  - `Ctrl+Enter`: Send message
  - `Ctrl+L`: Clear chat
  - `Ctrl+S`: Save chat
  - `Ctrl+O`: Load chat
- Export chats to Markdown/JSON
- Real-time status updates
- Progress indicators

## Prerequisites
- Windows 11 
- Python 3.8+
- Ollama installed and running
- Git (optional)

## Installation

1. Clone the repository or download the source code:
```bash
git clone <repository-url>
cd ollama-chat-gui
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure Ollama is running on localhost:11434

5. Run the application:
```bash
python ollama_chatbot.py
```

## Model Presets

The application comes with predefined model presets for different conversation styles:

- **Balanced** (Default)
  - Temperature: 0.7
  - Top P: 0.9
  - Top K: 40

- **Creative**
  - Temperature: 0.9
  - Top P: 0.95
  - Top K: 100

- **Precise**
  - Temperature: 0.3
  - Top P: 0.85
  - Top K: 20

- **Custom**
  - Fully configurable parameters

## Usage Tips

1. **System Prompt**: Customize the AI's behavior by setting a system prompt
2. **Context Management**: Monitor and clear conversation context as needed
3. **Model Parameters**: Use presets or customize for different conversation styles
4. **Code Blocks**: The chat supports markdown and code syntax highlighting
5. **Chat Export**: Export conversations in markdown format with formatting preserved

## Dependencies
- PyQt5: GUI framework
- requests: API communication
- markdown: Text formatting
- Pygments: Syntax highlighting
- rich: Enhanced text rendering

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
