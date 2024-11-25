import sys
import json
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                            QComboBox, QLabel, QMessageBox, QProgressBar, 
                            QSlider, QDialog, QListWidget, QFileDialog,
                            QStatusBar, QShortcut, QSpinBox, QMenu, QInputDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QKeySequence, QTextCharFormat, QSyntaxHighlighter, QClipboard
import requests
import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

# Model presets for different conversation styles
MODEL_PRESETS = {
    "Balanced": {"temperature": 0.7, "top_p": 0.9, "top_k": 40},
    "Creative": {"temperature": 0.9, "top_p": 0.95, "top_k": 100},
    "Precise": {"temperature": 0.3, "top_p": 0.85, "top_k": 20},
    "Custom": {"temperature": 0.7, "top_p": 0.9, "top_k": 40}
}

class MessageStyle:
    USER_COLOR = "#4CAF50"  # Green
    AI_COLOR = "#2196F3"    # Blue
    CODE_BG = "#1E1E1E"     # Dark background for code
    ERROR_COLOR = "#F44336" # Red for errors
    
    @staticmethod
    def format_message(role, content):
        color = MessageStyle.USER_COLOR if role == "user" else MessageStyle.AI_COLOR
        return (f'<div style="margin: 10px 0; padding: 10px; border-radius: 5px; '
                f'background-color: rgba({",".join(str(int(color[i:i+2], 16)) for i in (1,3,5))}, 0.1);">'
                f'<span style="color: {color}; font-weight: bold;">{role.title()}: </span>{content}</div>')
    
    @staticmethod
    def format_code(code, language=""):
        return (f'<pre style="background-color: {MessageStyle.CODE_BG}; padding: 10px; '
                f'border-radius: 5px; margin: 10px 0;"><code>{code}</code>'
                f'<button class="copy-btn">Copy</button></pre>')
    
    @staticmethod
    def format_error(message):
        return (f'<div style="color: {MessageStyle.ERROR_COLOR}; margin: 10px 0; padding: 10px; '
                f'border-left: 3px solid {MessageStyle.ERROR_COLOR};">{message}</div>')

class MarkdownHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.code_format = QTextCharFormat()
        self.code_format.setBackground(QColor("#2d2d2d"))
        self.code_format.setForeground(QColor("#d4d4d4"))
        self.code_format.setFontFamily("Consolas")

    def highlightBlock(self, text):
        if text.startswith("```") or text.startswith("    "):
            self.setFormat(0, len(text), self.code_format)

class ChatWorker(QThread):
    response_received = pyqtSignal(str)
    chunk_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model, message, history, base_url):
        super().__init__()
        self.model = model
        self.message = message
        self.history = history
        self.base_url = base_url
        self.full_response = ""
        self.chunk_buffer = ""
        self.is_first_chunk = True

    def should_emit_buffer(self):
        # Don't break if buffer is too small
        if len(self.chunk_buffer.strip()) < 5:
            return False
            
        # Always emit first chunk immediately for responsiveness
        if self.is_first_chunk and len(self.chunk_buffer.strip()) > 0:
            self.is_first_chunk = False
            return True
            
        # Look for sentence endings
        for end in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            if end in self.chunk_buffer:
                # Split at the last sentence ending
                last_end = max(self.chunk_buffer.rfind(end) for end in ['. ', '! ', '? ', '.\n', '!\n', '?\n'])
                if last_end != -1:
                    self.chunk_buffer = self.chunk_buffer[:last_end + 2]
                    return True
        
        # Emit on long enough content with proper breaks
        if len(self.chunk_buffer) > 40:
            # Try to break at the last punctuation or space
            for break_char in [',', ';', ':', ' ']:
                last_break = self.chunk_buffer.rfind(break_char, 30)
                if last_break != -1:
                    self.chunk_buffer = self.chunk_buffer[:last_break + 1]
                    return True
        
        return False

    def run(self):
        try:
            payload = {
                "model": self.model,
                "messages": self.history + [
                    {"role": "user", "content": self.message}
                ],
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }

            with requests.post(
                f"{self.base_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=30
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if chunk.get('done', False):
                                if self.chunk_buffer.strip():
                                    self.chunk_received.emit(self.chunk_buffer)
                                break
                            content = chunk.get('message', {}).get('content', '')
                            if content:
                                self.full_response += content
                                self.chunk_buffer += content
                                
                                if self.should_emit_buffer():
                                    self.chunk_received.emit(self.chunk_buffer)
                                    self.chunk_buffer = ""
                                    
                        except json.JSONDecodeError:
                            continue

            self.response_received.emit(self.full_response)

        except Exception as e:
            self.error_occurred.emit(str(e))

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.base_url = "http://localhost:11434/api"
        self.chat_history = []
        self.chat_log_dir = "chat_logs"
        self.current_model = "llama3"
        self.system_prompt = "You are a helpful AI assistant."
        self.preset_name = "Balanced"
        self.load_preset(self.preset_name)
        self.max_tokens = 4096
        self.estimated_tokens = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.dark_mode = True
        self.conversation_branches = []
        self.current_branch = []
        os.makedirs(self.chat_log_dir, exist_ok=True)
        
        self.init_ui()
        self.setup_shortcuts()
        self.check_ollama_status()
        self.load_available_models()
        
        # Setup status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.clear_status)
        self.status_timer.setSingleShot(True)

    def load_preset(self, preset_name):
        preset = MODEL_PRESETS[preset_name]
        self.temperature = preset["temperature"]
        self.top_p = preset["top_p"]
        self.top_k = preset["top_k"]
        
    def init_ui(self):
        self.setWindowTitle('Ollama Chat')
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 8px;
                color: #ffffff;
            }
            QLineEdit {
                padding: 8px;
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                color: #ffffff;
            }
            QPushButton {
                padding: 8px 16px;
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QComboBox {
                padding: 8px;
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Model selection and parameters
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItem(self.current_model)
        
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Preset:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(MODEL_PRESETS.keys())
        self.preset_combo.setCurrentText(self.preset_name)
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        
        temp_label = QLabel("Temperature:")
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(int(self.temperature * 100))
        self.temp_slider.valueChanged.connect(self.update_temperature)
        self.temp_value = QLabel(f"{self.temperature:.2f}")
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(temp_label)
        preset_layout.addWidget(self.temp_slider)
        preset_layout.addWidget(self.temp_value)
        preset_layout.addStretch()
        layout.addLayout(model_layout)
        layout.addLayout(preset_layout)

        # System prompt
        system_layout = QHBoxLayout()
        system_label = QLabel("System Prompt:")
        self.system_input = QLineEdit(self.system_prompt)
        system_layout.addWidget(system_label)
        system_layout.addWidget(self.system_input)
        layout.addLayout(system_layout)

        # Context visualization
        context_layout = QHBoxLayout()
        self.context_label = QLabel("Context: 0 messages")
        self.context_clear = QPushButton("Clear Context")
        self.context_clear.clicked.connect(self.clear_context)
        context_layout.addWidget(self.context_label)
        context_layout.addWidget(self.context_clear)
        context_layout.addStretch()
        layout.addLayout(context_layout)

        # Search bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in chat...")
        self.search_input.returnPressed.connect(self.search_chat)
        self.search_prev = QPushButton("↑")
        self.search_next = QPushButton("↓")
        self.search_prev.clicked.connect(lambda: self.search_chat(direction=-1))
        self.search_next.clicked.connect(lambda: self.search_chat(direction=1))
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_prev)
        search_layout.addWidget(self.search_next)
        layout.addLayout(search_layout)

        # Branch controls
        branch_layout = QHBoxLayout()
        self.branch_label = QLabel("Current Branch: Main")
        self.new_branch_btn = QPushButton("New Branch")
        self.new_branch_btn.clicked.connect(self.create_new_branch)
        self.switch_branch_btn = QPushButton("Switch Branch")
        self.switch_branch_btn.clicked.connect(self.switch_branch)
        branch_layout.addWidget(self.branch_label)
        branch_layout.addWidget(self.new_branch_btn)
        branch_layout.addWidget(self.switch_branch_btn)
        layout.addLayout(branch_layout)

        # Chat display with markdown support
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        self.chat_display.setMinimumHeight(400)
        self.markdown_highlighter = MarkdownHighlighter(self.chat_display.document())
        layout.addWidget(self.chat_display)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumHeight(2)
        layout.addWidget(self.progress)

        # Token estimation
        token_layout = QHBoxLayout()
        self.token_label = QLabel("Estimated Tokens: 0 / 4096")
        token_layout.addWidget(self.token_label)
        token_layout.addStretch()
        layout.addLayout(token_layout)

        # Typing indicator
        self.typing_label = QLabel("")
        self.typing_label.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(self.typing_label)

        # Input area
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Chat")
        self.clear_button.clicked.connect(self.clear_chat)
        self.save_button = QPushButton("Save Chat")
        self.save_button.clicked.connect(self.save_chat)
        self.load_button = QPushButton("Load Chat")
        self.load_button.clicked.connect(self.load_chat)
        self.export_button = QPushButton("Export Chat")
        self.export_button.clicked.connect(self.export_chat)
        self.theme_button = QPushButton("")
        self.theme_button.setToolTip("Toggle Dark/Light Theme")
        self.theme_button.clicked.connect(self.toggle_theme)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.theme_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Character count
        self.char_count_label = QLabel("Characters: 0")
        layout.addWidget(self.char_count_label)

        # Add status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def setup_shortcuts(self):
        # Send message shortcut (Ctrl+Enter)
        self.send_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.send_shortcut.activated.connect(self.send_message)
        
        # Clear chat shortcut (Ctrl+L)
        self.clear_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self.clear_shortcut.activated.connect(self.clear_chat)
        
        # Save chat shortcut (Ctrl+S)
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.save_chat)
        
        # Load chat shortcut (Ctrl+O)
        self.load_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        self.load_shortcut.activated.connect(self.load_chat)

    def on_preset_changed(self, preset_name):
        self.preset_name = preset_name
        self.load_preset(preset_name)
        if preset_name != "Custom":
            self.temp_slider.setValue(int(self.temperature * 100))
            self.update_context_info()
            
    def clear_context(self):
        self.chat_history = []
        self.update_context_info()
        self.set_status("Context cleared")
        
    def update_context_info(self):
        msg_count = len(self.chat_history)
        tokens = sum(len(msg["content"].split()) for msg in self.chat_history)
        self.context_label.setText(f"Context: {msg_count} messages ({tokens} est. tokens)")
        
    def estimate_tokens(self, text):
        # Simple estimation: ~4 characters per token
        self.estimated_tokens = len(text) // 4
        self.token_label.setText(f"Estimated Tokens: {self.estimated_tokens} / {self.max_tokens}")
        return self.estimated_tokens

    def set_status(self, message, timeout=3000):
        self.statusBar.showMessage(message)
        self.status_timer.start(timeout)

    def clear_status(self):
        self.statusBar.showMessage("Ready")

    def export_chat(self):
        if not self.chat_history:
            self.set_status("No chat history to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Chat",
            os.path.join(os.path.expanduser("~"), "chat_export.md"),
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Chat Export\n\n")
                f.write(f"Model: {self.current_model}\n")
                f.write(f"Temperature: {self.temperature}\n")
                f.write(f"System Prompt: {self.system_input.text()}\n\n")
                f.write("---\n\n")
                
                for msg in self.chat_history:
                    role = "You" if msg["role"] == "user" else "AI"
                    f.write(f"### {role}:\n{msg['content']}\n\n")
            
            self.set_status(f"Chat exported to: {file_path}")
        except Exception as e:
            self.set_status(f"Export failed: {str(e)}")

    def check_ollama_status(self):
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama server is not responding correctly")
            self.reconnect_attempts = 0
        except Exception as e:
            self.reconnect_attempts += 1
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.set_status(f"Reconnecting to Ollama (attempt {self.reconnect_attempts})...")
                QTimer.singleShot(2000, self.check_ollama_status)
            else:
                QMessageBox.critical(self, "Error", 
                                   "Could not connect to Ollama server. Please make sure it's running.")
                sys.exit(1)

    def load_available_models(self):
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                self.model_combo.clear()
                for model in models['models']:
                    self.model_combo.addItem(model['name'])
        except Exception as e:
            QMessageBox.warning(self, "Warning", 
                              "Could not load available models. Using default model.")

    def update_temperature(self):
        self.temperature = self.temp_slider.value() / 100
        self.temp_value.setText(f"{self.temperature:.2f}")

    def send_message(self):
        message = self.message_input.text().strip()
        if not message:
            return

        # Check token limit
        if self.estimate_tokens(message) > self.max_tokens:
            error_msg = MessageStyle.format_error("Message exceeds token limit")
            self.chat_display.append(error_msg)
            self.set_status("Message too long")
            return

        # Display user message with styling
        formatted_message = MessageStyle.format_message("user", message)
        self.chat_display.append(formatted_message)
        self.message_input.clear()
        self.message_input.setEnabled(False)
        self.send_button.setEnabled(False)
        
        # Show progress and typing indicator
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.typing_label.setText("AI is typing...")
        self.set_status("Generating response...")

        # Update character count
        self.update_char_count(message)

        # Create worker thread with system prompt
        history = []
        if self.system_input.text().strip():
            history.append({"role": "system", "content": self.system_input.text().strip()})
        history.extend(self.chat_history)
        
        self.worker = ChatWorker(
            self.model_combo.currentText(),
            message,
            history,
            self.base_url
        )
        self.worker.chunk_received.connect(self.handle_chunk)
        self.worker.response_received.connect(self.handle_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def update_char_count(self, message):
        self.char_count_label.setText(f"Characters: {len(message)}")

    def handle_chunk(self, chunk):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        
        # Initialize AI response if needed
        if cursor.position() == 0 or not self.chat_display.toPlainText().rstrip().endswith("AI:"):
            formatted_message = MessageStyle.format_message("assistant", "")
            cursor.insertHtml(formatted_message)
            cursor.movePosition(cursor.End)
        
        # Handle code blocks
        if "```" in chunk:
            code_block = chunk.split("```")[1]
            language = code_block.split("\n")[0].strip()
            code = "\n".join(code_block.split("\n")[1:])
            formatted_code = MessageStyle.format_code(code, language)
            cursor.insertHtml(formatted_code)
        else:
            cursor.insertText(chunk.strip())
            
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def handle_response(self, response):
        # Clean up any trailing whitespace or newlines
        current_text = self.chat_display.toPlainText().rstrip()
        if not current_text.endswith(response.strip()):
            cursor = self.chat_display.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText("\n")
            
        self.chat_history.extend([
            {"role": "user", "content": self.message_input.text().strip()},
            {"role": "assistant", "content": response}
        ])
        self.set_status("Response received")
        self.typing_label.clear()

    def handle_error(self, error_message):
        QMessageBox.warning(self, "Error", f"An error occurred: {error_message}")
        self.set_status(f"Error: {error_message}")
        self.typing_label.clear()

    def on_worker_finished(self):
        self.message_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.progress.setVisible(False)
        self.message_input.setFocus()
        self.typing_label.clear()

    def clear_chat(self):
        self.chat_display.clear()
        self.chat_history = []

    def save_chat(self):
        if not self.chat_history:
            QMessageBox.information(self, "Info", "No chat history to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_log_{timestamp}.json"
        filepath = os.path.join(self.chat_log_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, indent=2)
            QMessageBox.information(self, "Success", f"Chat saved to: {filepath}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save chat: {str(e)}")

    def load_chat(self):
        files = [f for f in os.listdir(self.chat_log_dir) if f.endswith('.json')]
        if not files:
            QMessageBox.information(self, "Info", "No chat logs found.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Load Chat")
        layout = QVBoxLayout()
        
        list_widget = QListWidget()
        list_widget.addItems(files)
        layout.addWidget(list_widget)
        
        load_button = QPushButton("Load Selected")
        layout.addWidget(load_button)
        
        dialog.setLayout(layout)
        
        def load_selected():
            selected = list_widget.currentItem()
            if selected:
                try:
                    filepath = os.path.join(self.chat_log_dir, selected.text())
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.chat_history = json.load(f)
                    
                    self.chat_display.clear()
                    for msg in self.chat_history:
                        role = "You" if msg["role"] == "user" else "AI"
                        self.chat_display.append(f"\n{role}: {msg['content']}")
                    
                    dialog.accept()
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not load chat: {str(e)}")
        
        load_button.clicked.connect(load_selected)
        dialog.exec_()

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.theme_button.setText("" if self.dark_mode else "")
        
        # Update color scheme
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; }
                QTextEdit, QLineEdit, QComboBox {
                    background-color: #2d2d2d;
                    border: 1px solid #3d3d3d;
                    border-radius: 4px;
                    padding: 8px;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                }
                QPushButton:hover { background-color: #106ebe; }
                QPushButton:pressed { background-color: #005a9e; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget { background-color: #ffffff; color: #000000; }
                QTextEdit, QLineEdit, QComboBox {
                    background-color: #f5f5f5;
                    border: 1px solid #dddddd;
                    border-radius: 4px;
                    padding: 8px;
                    color: #000000;
                }
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                }
                QPushButton:hover { background-color: #106ebe; }
                QPushButton:pressed { background-color: #005a9e; }
            """)

    def copy_code_to_clipboard(self, code):
        clipboard = QApplication.clipboard()
        clipboard.setText(code)
        self.set_status("Code copied to clipboard")

    def handle_copy_button_click(self, button):
        # Find the code block associated with this button
        code_block = button.parent()
        if code_block and isinstance(code_block, QTextEdit):
            code = code_block.toPlainText()
            self.copy_code_to_clipboard(code)

    def search_chat(self, direction=1):
        query = self.search_input.text().lower()
        if not query:
            return
            
        cursor = self.chat_display.textCursor()
        current_pos = cursor.position()
        
        # Get all text
        text = self.chat_display.toPlainText().lower()
        
        if direction > 0:
            # Search forward
            next_pos = text.find(query, current_pos)
            if next_pos == -1:  # Wrap around
                next_pos = text.find(query)
        else:
            # Search backward
            next_pos = text.rfind(query, 0, current_pos)
            if next_pos == -1:  # Wrap around
                next_pos = text.rfind(query)
                
        if next_pos != -1:
            cursor.setPosition(next_pos)
            cursor.movePosition(cursor.Right, cursor.KeepAnchor, len(query))
            self.chat_display.setTextCursor(cursor)
            self.chat_display.ensureCursorVisible()
        else:
            self.set_status("Text not found")

    def create_new_branch(self):
        name, ok = QInputDialog.getText(self, "New Branch", "Enter branch name:")
        if ok and name:
            # Save current conversation state
            branch = {
                "name": name,
                "history": self.chat_history.copy(),
                "system_prompt": self.system_input.text(),
                "model": self.model_combo.currentText(),
                "temperature": self.temperature
            }
            self.conversation_branches.append(branch)
            self.current_branch = branch
            self.branch_label.setText(f"Current Branch: {name}")
            self.set_status(f"Created new branch: {name}")

    def switch_branch(self):
        if not self.conversation_branches:
            self.set_status("No branches available")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Switch Branch")
        layout = QVBoxLayout(dialog)
        
        branch_list = QListWidget()
        branch_list.addItem("Main")
        for branch in self.conversation_branches:
            branch_list.addItem(branch["name"])
            
        layout.addWidget(branch_list)
        
        def switch():
            selected = branch_list.currentItem()
            if selected:
                branch_name = selected.text()
                if branch_name == "Main":
                    self.current_branch = None
                    self.chat_history = []
                else:
                    for branch in self.conversation_branches:
                        if branch["name"] == branch_name:
                            self.current_branch = branch
                            self.chat_history = branch["history"].copy()
                            self.system_input.setText(branch["system_prompt"])
                            self.model_combo.setCurrentText(branch["model"])
                            self.temperature = branch["temperature"]
                            self.temp_slider.setValue(int(self.temperature * 100))
                            break
                            
                self.branch_label.setText(f"Current Branch: {branch_name}")
                self.update_chat_display()
                dialog.accept()
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(switch)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.exec_()

    def update_chat_display(self):
        self.chat_display.clear()
        for msg in self.chat_history:
            formatted_msg = MessageStyle.format_message(msg["role"], msg["content"])
            self.chat_display.append(formatted_msg)

def main():
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
