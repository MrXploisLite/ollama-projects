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
import subprocess
import time
import threading

# Model presets for different conversation styles
MODEL_PRESETS = {
    "Balanced": {"temperature": 0.7, "top_p": 0.9, "top_k": 40},
    "Creative": {"temperature": 0.9, "top_p": 0.95, "top_k": 100},
    "Precise": {"temperature": 0.3, "top_p": 0.85, "top_k": 20},
    "Custom": {"temperature": 0.7, "top_p": 0.9, "top_k": 40}
}

class ServerStatus:
    CONNECTED = "Connected"
    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting..."
    ERROR = "Error"

class MessageStyle:
    USER_COLOR = "#4CAF50"  # Green
    AI_COLOR = "#2196F3"    # Blue
    CODE_BG = "#1E1E1E"     # Dark background for code
    ERROR_COLOR = "#F44336" # Red for errors
    
    @staticmethod
    def format_message(role, content):
        """Format a message with role-based styling and markdown/code processing."""
        # Process the content for markdown and code blocks
        formatted_content = content
        
        # Handle code blocks first
        if "```" in formatted_content:
            parts = formatted_content.split("```")
            formatted_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Not a code block
                    formatted_parts.append(MessageStyle.process_markdown(part))
                else:  # Code block
                    try:
                        language = part.split('\n')[0].strip()
                        code = '\n'.join(part.split('\n')[1:])
                        formatted_parts.append(MessageStyle.format_code(code, language))
                    except IndexError:
                        formatted_parts.append(MessageStyle.format_code(part, ""))
            formatted_content = "".join(formatted_parts)
        else:
            formatted_content = MessageStyle.process_markdown(formatted_content)
        
        # Apply role-based styling
        color = MessageStyle.USER_COLOR if role.lower() == "user" else MessageStyle.AI_COLOR
        return (f'<div style="margin: 10px 0; padding: 10px; border-radius: 5px; '
                f'background-color: rgba({",".join(str(int(color[i:i+2], 16)) for i in (1,3,5))}, 0.1);">'
                f'<span style="color: {color}; font-weight: bold;">{role.title()}: </span>'
                f'{formatted_content}</div>')
    
    @staticmethod
    def process_markdown(text):
        """Process markdown text to HTML, excluding code blocks."""
        try:
            # Convert markdown to HTML
            html = markdown.markdown(text, extensions=['fenced_code', 'tables'])
            
            # Add custom styling
            html = html.replace('<p>', '<p style="margin: 5px 0;">')
            html = html.replace('<ul>', '<ul style="margin: 5px 0; padding-left: 20px;">')
            html = html.replace('<ol>', '<ol style="margin: 5px 0; padding-left: 20px;">')
            html = html.replace('<li>', '<li style="margin: 2px 0;">')
            
            return html
        except Exception:
            return text
    
    @staticmethod
    def format_code(code, language=""):
        """Format code with syntax highlighting and copy button."""
        try:
            if language and language != "":
                lexer = get_lexer_by_name(language, stripall=True)
            else:
                from pygments.lexers import guess_lexer
                try:
                    lexer = guess_lexer(code)
                except:
                    lexer = get_lexer_by_name("text")
            
            formatter = HtmlFormatter(style='monokai', 
                                   cssclass='highlight',
                                   noclasses=True,
                                   nowrap=True)
            highlighted = highlight(code, lexer, formatter)
        except Exception:
            # Fallback to simple formatting if highlighting fails
            highlighted = f'<pre style="color: #f8f8f2;">{code}</pre>'
        
        return (f'<div style="background-color: {MessageStyle.CODE_BG}; padding: 10px; '
                f'border-radius: 5px; margin: 10px 0; position: relative;">'
                f'{highlighted}'
                f'<button onclick="copyCode(this)" style="position: absolute; top: 5px; right: 5px; '
                f'padding: 5px 10px; background: #333; color: white; border: none; '
                f'border-radius: 3px; cursor: pointer;">Copy</button></div>')
    
    @staticmethod
    def format_error(message):
        """Format error messages with distinctive styling."""
        return (f'<div style="color: {MessageStyle.ERROR_COLOR}; margin: 10px 0; padding: 10px; '
                f'background-color: rgba({",".join(str(int(MessageStyle.ERROR_COLOR[i:i+2], 16)) for i in (1,3,5))}, 0.1); '
                f'border-left: 3px solid {MessageStyle.ERROR_COLOR}; border-radius: 3px;">'
                f'<span style="font-weight: bold;">Error: </span>{message}</div>')

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

class GitAutoSync:
    def __init__(self, repo_path, interval=300):  # 5 minutes default
        self.repo_path = repo_path
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _sync_loop(self):
        while self.running:
            try:
                self._perform_sync()
            except Exception as e:
                print(f"Auto-sync error: {str(e)}")
            time.sleep(self.interval)

    def _perform_sync(self):
        try:
            # Add all changes
            subprocess.run(["git", "add", "."], cwd=self.repo_path, check=True)
            
            # Create commit with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            subprocess.run(
                ["git", "commit", "-m", f"Auto-sync: {timestamp}"],
                cwd=self.repo_path, check=True
            )
            
            # Push changes
            subprocess.run(["git", "push", "origin", "main"], cwd=self.repo_path, check=True)
        except subprocess.CalledProcessError:
            # If nothing to commit, just skip
            pass

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
        
    def run(self):
        try:
            payload = {
                "model": self.model,
                "messages": self.history + [{"role": "user", "content": self.message}],
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 4096
                }
            }
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                stream=True,
                timeout=60
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        if 'message' in json_response and 'content' in json_response['message']:
                            content = json_response['message']['content']
                            self.full_response += content
                            self.chunk_received.emit(content)
                    except json.JSONDecodeError:
                        continue
                        
            if self.full_response:
                self.response_received.emit(self.full_response)
                
        except Exception as e:
            self.error_occurred.emit(str(e))

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.base_url = "http://localhost:11434/api"
        self.chat_history = []
        self.chat_log_dir = "chat_logs"
        self.current_model = "llama2"
        self.system_prompt = "You are a helpful AI assistant."
        self.preset_name = "Balanced"
        self.server_status = ServerStatus.DISCONNECTED
        self.load_preset(self.preset_name)
        
        # Setup server check timer
        self.server_check_timer = QTimer(self)
        self.server_check_timer.timeout.connect(self.check_ollama_status)
        self.server_check_timer.start(30000)  # Check every 30 seconds
        
        # Initialize Git auto-sync
        self.git_sync = GitAutoSync(os.path.dirname(os.path.abspath(__file__)))
        self.git_sync.start()
        
        # Create chat logs directory
        os.makedirs(self.chat_log_dir, exist_ok=True)
        
        # Setup status update timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.clear_status)
        self.status_timer.setSingleShot(True)
        
        self.init_ui()
        self.setup_shortcuts()
        self.check_ollama_status()
        self.load_available_models()

    def load_preset(self, preset_name):
        preset = MODEL_PRESETS[preset_name]
        self.temperature = preset["temperature"]
        self.top_p = preset["top_p"]
        self.top_k = preset["top_k"]
        
    def init_ui(self):
        self.setWindowTitle("Ollama Chat")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
            }
            QTextEdit, QLineEdit {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border: 1px solid #3E3E3E;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1084D9;
            }
            QComboBox {
                background-color: #2D2D2D;
                color: #FFFFFF;
                border: 1px solid #3E3E3E;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Top bar with status and model selection
        top_bar = QHBoxLayout()
        
        # Status indicator
        status_layout = QHBoxLayout()
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(12, 12)
        status_text = QLabel("Server Status")
        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(status_text)
        top_bar.addLayout(status_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(150)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        top_bar.addStretch()
        top_bar.addLayout(model_layout)
        
        layout.addLayout(top_bar)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(400)
        layout.addWidget(self.chat_display)

        # Input area
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setFixedWidth(100)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3E3E3E;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078D4;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        # Add JavaScript for code copying
        copy_script = """
            <script>
            function copyCode(button) {
                var codeBlock = button.parentElement;
                var code = codeBlock.querySelector('pre').innerText;
                navigator.clipboard.writeText(code).then(function() {
                    button.textContent = 'Copied!';
                    setTimeout(function() {
                        button.textContent = 'Copy';
                    }, 2000);
                }).catch(function(err) {
                    console.error('Failed to copy:', err);
                    button.textContent = 'Error';
                });
            }
            </script>
        """
        self.chat_display.document().setHtml(copy_script)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        self.chat_display.setMinimumHeight(400)
        layout.addWidget(self.chat_display)

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
        self.token_label.setText(f"Estimated Tokens: {self.estimated_tokens} / 4096")
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
        """Check if Ollama server is running and update status."""
        try:
            self.update_server_status(ServerStatus.CONNECTING)
            response = requests.get(f"{self.base_url}/tags")
            if response.status_code == 200:
                self.update_server_status(ServerStatus.CONNECTED)
                self.load_available_models()
                self.reconnect_attempts = 0
                return True
        except requests.exceptions.RequestException as e:
            self.update_server_status(ServerStatus.ERROR)
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_attempts += 1
                self.set_status(f"Connection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                QTimer.singleShot(5000, self.check_ollama_status)
            else:
                self.set_status("Could not connect to Ollama server. Please check if it's running.")
        return False

    def load_available_models(self):
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                self.model_combo.clear()
                if 'models' in models:
                    for model in models['models']:
                        self.model_combo.addItem(model['name'])
                    if self.model_combo.count() > 0:
                        self.current_model = self.model_combo.itemText(0)
                        self.set_status(f"Loaded {self.model_combo.count()} models")
                else:
                    self.model_combo.addItem(self.current_model)
                    self.set_status("No models found, using default model")
        except Exception as e:
            self.model_combo.clear()
            self.model_combo.addItem(self.current_model)
            self.set_status("Could not load models, using default model")

    def update_temperature(self):
        self.temperature = self.temp_slider.value() / 100
        self.temp_value.setText(f"{self.temperature:.2f}")

    def send_message(self):
        message = self.message_input.text().strip()
        if not message:
            return

        # Check token limit
        if self.estimate_tokens(message) > 4096:
            error_msg = MessageStyle.format_error("Message exceeds token limit")
            self.chat_display.append(error_msg)
            self.set_status("Message too long")
            return

        # Display user message
        formatted_message = MessageStyle.format_message("User", message)
        self.chat_display.append(formatted_message)
        self.message_input.clear()
        self.message_input.setEnabled(False)
        self.send_button.setEnabled(False)

        # Add user message to history
        self.chat_history.append({"role": "user", "content": message})

        # Initialize AI response
        self.chat_display.append(MessageStyle.format_message("Assistant", ""))

        # Create and start worker thread
        history = []
        if self.system_input.text().strip():
            history.append({"role": "system", "content": self.system_input.text().strip()})
        history.extend(self.chat_history[:-1])  # Exclude the last message as it's added in the worker

        self.worker = ChatWorker(
            self.current_model,
            message,
            history,
            self.base_url
        )
        self.worker.chunk_received.connect(self.handle_chunk)
        self.worker.response_received.connect(self.handle_response)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def handle_chunk(self, chunk):
        """Handle incoming message chunks with proper formatting."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        
        # If this is a new response, ensure we're on a new line
        if not cursor.block().text().strip():
            cursor.insertBlock()
        
        # Process the chunk for any code blocks or markdown
        formatted_chunk = chunk
        
        # If chunk contains a complete code block, format it
        if "```" in chunk:
            parts = chunk.split("```")
            formatted_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Not a code block
                    formatted_parts.append(MessageStyle.process_markdown(part))
                else:  # Code block
                    try:
                        language = part.split('\n')[0].strip()
                        code = '\n'.join(part.split('\n')[1:])
                        formatted_parts.append(MessageStyle.format_code(code, language))
                    except IndexError:
                        formatted_parts.append(MessageStyle.format_code(part, ""))
            formatted_chunk = "".join(formatted_parts)
        else:
            # For regular text, just insert it as is
            formatted_chunk = chunk
        
        cursor.insertHtml(formatted_chunk)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()
        QApplication.processEvents()

    def handle_response(self, response):
        # Add the complete message to chat history
        if response.strip():
            self.chat_history.append({"role": "assistant", "content": response})
        self.message_input.setEnabled(True)
        self.send_button.setEnabled(True)

    def handle_error(self, error_message):
        error_msg = MessageStyle.format_error(f"Error: {error_message}")
        self.chat_display.append(error_msg)
        self.message_input.setEnabled(True)
        self.send_button.setEnabled(True)

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

    def update_server_status(self, status):
        """Update server status and UI indicators."""
        self.server_status = status
        status_color = {
            ServerStatus.CONNECTED: "#4CAF50",    # Green
            ServerStatus.DISCONNECTED: "#F44336", # Red
            ServerStatus.CONNECTING: "#FFC107",   # Yellow
            ServerStatus.ERROR: "#F44336"         # Red
        }.get(status, "#F44336")
        
        self.status_indicator.setStyleSheet(f"""
            QLabel {{
                color: {status_color};
                padding: 5px;
                border-radius: 10px;
                background: qradialgradient(cx:0.5, cy:0.5, radius: 0.5, fx:0.5, fy:0.5,
                    stop:0 {status_color}, stop:0.5 {status_color}, stop:0.6 transparent);
            }}
        """)
        self.status_indicator.setToolTip(f"Server Status: {status}")
        
        # Update UI elements based on status
        is_connected = status == ServerStatus.CONNECTED
        self.message_input.setEnabled(is_connected)
        self.send_button.setEnabled(is_connected)
        self.model_combo.setEnabled(is_connected)
        
        if not is_connected:
            self.set_status(f"Server {status}")

def main():
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
