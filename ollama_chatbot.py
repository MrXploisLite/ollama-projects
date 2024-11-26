import sys
import json
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                            QComboBox, QLabel, QMessageBox, QProgressBar, 
                            QSlider, QDialog, QListWidget, QFileDialog,
                            QStatusBar, QShortcut, QSpinBox, QMenu, QInputDialog,
                            QDoubleSpinBox, QGroupBox, QFormLayout, QListWidgetItem,
                            QActionGroup, QColorDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import (QFont, QPalette, QColor, QKeySequence, QTextCharFormat, 
                      QSyntaxHighlighter, QClipboard)
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
    "Balanced": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "presence_penalty": 0,
        "frequency_penalty": 0
    },
    "Creative": {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 100,
        "repeat_penalty": 1.05,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2
    },
    "Precise": {
        "temperature": 0.3,
        "top_p": 0.85,
        "top_k": 20,
        "repeat_penalty": 1.2,
        "presence_penalty": -0.1,
        "frequency_penalty": -0.1
    },
    "Custom": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "presence_penalty": 0,
        "frequency_penalty": 0
    }
}

CHAT_PERSONAS = {
    "Default": "You are a helpful AI assistant.",
    "Professional": "You are a professional AI assistant focused on providing clear, accurate, and well-structured responses.",
    "Teacher": "You are a patient and knowledgeable teacher who explains concepts clearly and provides examples.",
    "Programmer": "You are a skilled programmer who provides detailed technical explanations and code examples.",
    "Custom": "You are a helpful AI assistant."
}

class ServerStatus:
    CONNECTED = "Connected"
    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting..."
    ERROR = "Error"

class MessageStyle:
    USER_COLOR = "#2ecc71"  # Green
    AI_COLOR = "#3498db"    # Blue
    CODE_BG = "#1e1e1e"     # Darker background for code
    ERROR_COLOR = "#e74c3c" # Red
    
    @staticmethod
    def format_code_block(code, language='python'):
        """Format code blocks with syntax highlighting."""
        try:
            lexer = get_lexer_by_name(language)
            formatter = HtmlFormatter(
                style='monokai',
                noclasses=True,
                nobackground=True,
                linenos=True,
                cssclass="source",
                wrapcode=True
            )
            highlighted = highlight(code, lexer, formatter)
            
            # Add custom styling for better readability
            styled_code = f'''
            <div style="background-color: {MessageStyle.CODE_BG}; 
                        padding: 15px; 
                        border-radius: 8px; 
                        margin: 10px 0;
                        font-family: 'Consolas', 'Monaco', monospace;
                        font-size: 14px;
                        line-height: 1.4;
                        overflow-x: auto;
                        border: 1px solid #2d2d2d;">
                <div style="color: #f8f8f2;">
                    {highlighted}
                </div>
            </div>
            '''
            return styled_code
        except Exception as e:
            # Fallback formatting if syntax highlighting fails
            return f'''
            <pre style="background-color: {MessageStyle.CODE_BG}; 
                       color: #f8f8f2;
                       padding: 15px; 
                       border-radius: 8px; 
                       margin: 10px 0;
                       font-family: 'Consolas', 'Monaco', monospace;
                       font-size: 14px;
                       line-height: 1.4;
                       overflow-x: auto;
                       border: 1px solid #2d2d2d;">
                <code>{code}</code>
            </pre>
            '''

    @staticmethod
    def format_message(text, is_user=True):
        """Format chat messages with proper styling and code highlighting."""
        color = MessageStyle.USER_COLOR if is_user else MessageStyle.AI_COLOR
        
        # Convert markdown to HTML with proper extensions
        html = markdown.markdown(text, extensions=[
            'fenced_code',
            'codehilite',
            'tables',
            'nl2br'
        ])
        
        # Find and replace code blocks with syntax highlighted versions
        import re
        code_block_pattern = r'<pre><code.*?>(.*?)</code></pre>'
        
        def replace_code_block(match):
            code = match.group(1)
            # Unescape HTML entities
            code = code.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&').replace('&#39;', "'").replace('&quot;', '"')
            # Try to detect language from opening fence
            language = 'python'  # default to python
            return MessageStyle.format_code_block(code, language)
        
        html = re.sub(code_block_pattern, replace_code_block, html, flags=re.DOTALL)
        
        # Add message styling with improved readability
        styled_html = f'''
        <div style="color: {color}; 
                    margin-bottom: 15px;
                    font-family: 'Segoe UI', 'Arial', sans-serif;">
            <div style="padding: 10px;
                        border-radius: 8px;
                        line-height: 1.5;">
                {html}
            </div>
        </div>
        '''
        return styled_html

    @staticmethod
    def format_error(error_text):
        """Format error messages with improved visibility."""
        return f'''
        <div style="color: {MessageStyle.ERROR_COLOR};
                    margin: 10px 0;
                    padding: 10px;
                    background-color: rgba(231, 76, 60, 0.1);
                    border-left: 4px solid {MessageStyle.ERROR_COLOR};
                    border-radius: 4px;
                    font-family: 'Segoe UI', 'Arial', sans-serif;">
            <div style="font-weight: bold;">Error:</div>
            <div style="margin-top: 5px;">{error_text}</div>
        </div>
        '''

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
        self.last_sync_time = None

    def start(self):
        if not self.running:
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
                self.last_sync_time = datetime.now()
            except Exception as e:
                print(f"Auto-sync error: {str(e)}")
            time.sleep(self.interval)

    def _perform_sync(self):
        try:
            # Check if there are any changes
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            if status.stdout.strip():
                # Add all changes
                subprocess.run(
                    ["git", "add", "."],
                    cwd=self.repo_path,
                    check=True
                )
                
                # Create commit with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                subprocess.run(
                    ["git", "commit", "-m", f"Auto-sync: {timestamp}"],
                    cwd=self.repo_path,
                    check=True
                )
                
                # Push changes
                subprocess.run(
                    ["git", "push", "origin", "main"],
                    cwd=self.repo_path,
                    check=True
                )
                print(f"Auto-sync completed at {timestamp}")
                return True
        except subprocess.CalledProcessError as e:
            print(f"Git operation failed: {str(e)}")
        except Exception as e:
            print(f"Auto-sync error: {str(e)}")
        return False

    def get_status(self):
        if not self.running:
            return "Git Auto-sync: Disabled"
        if not self.last_sync_time:
            return "Git Auto-sync: Waiting for first sync..."
        
        time_diff = datetime.now() - self.last_sync_time
        minutes = int(time_diff.total_seconds() / 60)
        return f"Last auto-sync: {minutes} minutes ago"

class ModelManager:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"
        self.available_models = []
        self.downloading = False
        
    def get_available_models(self):
        try:
            response = requests.get(f"{self.base_url}/tags")
            if response.status_code == 200:
                models = response.json()['models']
                self.available_models = [model['name'] for model in models]
                return self.available_models
            return []
        except:
            return []
    
    def download_model(self, model_name, progress_callback=None):
        if self.downloading:
            return False
            
        self.downloading = True
        try:
            response = requests.post(
                f"{self.base_url}/pull",
                json={"name": model_name},
                stream=True
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to download model: {response.text}")
                
            total_size = 0
            downloaded_size = 0
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                data = json.loads(line)
                if "total" in data:
                    total_size = data["total"]
                if "completed" in data:
                    downloaded_size = data["completed"]
                    
                if total_size > 0 and progress_callback:
                    progress = int((downloaded_size / total_size) * 100)
                    progress_callback(progress)
                    
            self.downloading = False
            return True
            
        except Exception as e:
            self.downloading = False
            raise Exception(f"Error downloading model: {str(e)}")
            
    def download_model_original(self, model_name, progress_callback=None):
        if self.downloading:
            return False
            
        self.downloading = True
        try:
            url = f"{self.base_url}/pull"
            headers = {"Content-Type": "application/json"}
            data = {"name": model_name}
            
            with requests.post(url, headers=headers, json=data, stream=True) as response:
                if response.status_code != 200:
                    return False
                    
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            if 'error' in json_response:
                                return False
                            
                            if progress_callback and 'completed' in json_response:
                                progress = int(json_response['completed'])
                                progress_callback(progress)
                                
                        except json.JSONDecodeError:
                            continue
                            
            return True
        except:
            return False
        finally:
            self.downloading = False

class ChatWorker(QThread):
    response_received = pyqtSignal(str)
    chunk_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, model, message, history, base_url, system_prompt=None):
        super().__init__()
        self.model = model
        self.message = message
        self.history = history.copy()
        if system_prompt:
            self.history.insert(0, {"role": "system", "content": system_prompt})
        self.history.append({"role": "user", "content": message})
        self.base_url = base_url
        self.full_response = ""
        self._is_running = True
        
    def stop(self):
        self._is_running = False
        
    def run(self):
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "model": self.model,
                    "messages": self.history,
                    "stream": True
                },
                stream=True
            )
            
            if response.status_code != 200:
                error_msg = f"Error: Server returned status code {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"Error: {error_data['error']}"
                except:
                    pass
                self.error_occurred.emit(error_msg)
                return
                
            for line in response.iter_lines():
                if not self._is_running:
                    break
                    
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    if "error" in data:
                        self.error_occurred.emit(f"Error: {data['error']}")
                        return
                        
                    if "message" in data:
                        content = data["message"].get("content", "")
                        if content:
                            self.chunk_received.emit(content)
                            self.full_response += content
                except json.JSONDecodeError:
                    self.error_occurred.emit("Error: Invalid response from server")
                    return
                except Exception as e:
                    self.error_occurred.emit(f"Error: {str(e)}")
                    return
                    
            if self._is_running:
                self.response_received.emit(self.full_response)
                
        except requests.exceptions.ConnectionError:
            self.error_occurred.emit("Error: Could not connect to Ollama server")
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
            
    def run_original(self):
        try:
            url = f"{self.base_url}/chat"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model,
                "messages": self.history,
                "stream": True
            }
            
            with requests.post(url, headers=headers, json=data, stream=True) as response:
                if response.status_code != 200:
                    self.error_occurred.emit(f"Server error: {response.status_code}")
                    return
                    
                for line in response.iter_lines():
                    if not self._is_running:
                        break
                        
                    if not line:
                        continue
                        
                    try:
                        chunk_data = json.loads(line)
                        if 'error' in chunk_data:
                            self.error_occurred.emit(chunk_data['error'])
                            return
                            
                        if 'response' in chunk_data:
                            chunk = chunk_data['response']
                            self.full_response += chunk
                            self.chunk_received.emit(chunk)
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        self.error_occurred.emit(f"Error processing response: {str(e)}")
                        return
                        
                if self._is_running and self.full_response:
                    self.response_received.emit(self.full_response)
                    
        except requests.exceptions.RequestException as e:
            if self._is_running:
                self.error_occurred.emit(f"Connection error: {str(e)}")
        except Exception as e:
            if self._is_running:
                self.error_occurred.emit(f"Unexpected error: {str(e)}")

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.base_url = "http://localhost:11434/api"
        self.chat_history = []
        self.chat_log_dir = "chat_logs"
        self.current_model = "llama2"
        self.system_prompt = "You are a helpful AI assistant. The user's name is Romy Rianata."
        self.preset_name = "Balanced"
        self.server_status = ServerStatus.DISCONNECTED
        self.model_manager = ModelManager()
        self.worker = None
        self.user_name = "Romy Rianata"
        self.dark_mode = True
        self.model_preset = "Balanced"
        
        # Initialize Git auto-sync
        self.git_sync = GitAutoSync(os.path.dirname(os.path.abspath(__file__)))
        
        # Load settings
        self.load_settings()
        
        # Load preset and create directories
        self.load_preset(self.preset_name)
        os.makedirs(self.chat_log_dir, exist_ok=True)
        
        # Setup timers
        self.setup_timers()
        
        # Initialize UI
        self.init_ui()
        self.setup_shortcuts()
        self.check_ollama_status()
        self.load_available_models()
        
        # Start Git auto-sync
        self.git_sync.start()
        
        # Set window properties
        self.setWindowTitle(f"Ollama Chat - {self.user_name}")
        self.setMinimumSize(800, 600)
        
        # Send welcome message
        self.display_welcome_message()

    def load_settings(self):
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r') as f:
                    settings = json.load(f)
                    self.user_name = settings.get('user_name', self.user_name)
                    self.dark_mode = settings.get('dark_mode', self.dark_mode)
                    self.system_prompt = settings.get('system_prompt', self.system_prompt)
        except Exception as e:
            print(f"Error loading settings: {e}")

    def save_settings(self):
        try:
            settings = {
                'user_name': self.user_name,
                'dark_mode': self.dark_mode,
                'system_prompt': self.system_prompt,
                'model_preset': self.model_preset
            }
            with open('settings.json', 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def display_welcome_message(self):
        welcome_msg = f"Welcome back, {self.user_name}! I'm your AI assistant powered by {self.current_model}. How can I help you today?"
        self.chat_history.append({"role": "assistant", "content": welcome_msg})
        self.update_chat_display()

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        self.save_settings()

    def apply_theme(self, bg_color=None, text_color=None):
        if self.dark_mode:
            if bg_color is None:
                bg_color = "#1e1e1e"
            if text_color is None:
                text_color = "#d4d4d4"
        else:
            if bg_color is None:
                bg_color = "#ffffff"
            if text_color is None:
                text_color = "#000000"
        
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {bg_color};
                color: {text_color};
            }}
            QTextEdit, QLineEdit {{
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
            }}
            QComboBox, QPushButton {{
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 5px;
            }}
            QComboBox:hover, QPushButton:hover {{
                background-color: #3d3d3d;
            }}
            QLabel {{
                color: #d4d4d4;
            }}
        """)

    def closeEvent(self, event):
        """Clean up resources before closing."""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        self.git_sync.stop()  # Stop Git auto-sync
        event.accept()

    def setup_timers(self):
        # Server check timer
        self.server_check_timer = QTimer(self)
        self.server_check_timer.timeout.connect(self.check_ollama_status)
        self.server_check_timer.start(30000)  # Check every 30 seconds
        
        # Status update timer
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.clear_status)
        self.status_timer.setSingleShot(True)
        
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add menu bar
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_chat_action = file_menu.addAction('New Chat')
        new_chat_action.setShortcut('Ctrl+N')
        new_chat_action.triggered.connect(self.clear_chat)
        
        save_chat_action = file_menu.addAction('Save Chat')
        save_chat_action.setShortcut('Ctrl+S')
        save_chat_action.triggered.connect(self.save_chat)
        
        load_chat_action = file_menu.addAction('Load Chat')
        load_chat_action.setShortcut('Ctrl+O')
        load_chat_action.triggered.connect(self.load_chat)
        
        export_chat_action = file_menu.addAction('Export Chat')
        export_chat_action.setShortcut('Ctrl+E')
        export_chat_action.triggered.connect(self.export_chat)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('Exit')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        
        # Settings menu
        settings_menu = menubar.addMenu('Settings')
        
        # Theme submenu
        theme_menu = settings_menu.addMenu('Theme')
        
        theme_action = theme_menu.addAction('Toggle Dark/Light')
        theme_action.triggered.connect(self.toggle_theme)
        
        customize_colors_action = theme_menu.addAction('Customize Colors')
        customize_colors_action.triggered.connect(self.customize_colors)
        
        # Model settings submenu
        model_menu = settings_menu.addMenu('Model Settings')
        
        select_model_action = model_menu.addAction('Select Model')
        select_model_action.triggered.connect(self.select_model)
        
        edit_params_action = model_menu.addAction('Edit Parameters')
        edit_params_action.triggered.connect(self.edit_model_params)
        
        # Persona submenu
        persona_menu = settings_menu.addMenu('Chat Persona')
        self.persona_group = QActionGroup(self)
        
        for persona in CHAT_PERSONAS:
            action = persona_menu.addAction(persona)
            action.setCheckable(True)
            action.setChecked(persona == "Default")
            self.persona_group.addAction(action)
            action.triggered.connect(lambda checked, p=persona: self.change_persona(p))
        
        # User settings
        edit_name_action = settings_menu.addAction('Edit User Name')
        edit_name_action.triggered.connect(self.edit_user_name)
        
        edit_prompt_action = settings_menu.addAction('Edit System Prompt')
        edit_prompt_action.triggered.connect(self.edit_system_prompt)
        
        # Git sync settings
        git_menu = settings_menu.addMenu('Git Auto-sync')
        
        toggle_git_action = git_menu.addAction('Enable/Disable Auto-sync')
        toggle_git_action.triggered.connect(self.toggle_git_sync)
        
        set_interval_action = git_menu.addAction('Set Sync Interval')
        set_interval_action.triggered.connect(self.set_git_sync_interval)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)
        
        shortcuts_action = help_menu.addAction('Keyboard Shortcuts')
        shortcuts_action.triggered.connect(self.show_shortcuts)
        
        # Add chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setAcceptRichText(True)
        layout.addWidget(self.chat_display)
        
        # Add input area
        input_layout = QHBoxLayout()
        
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setMaximumHeight(100)
        self.message_input.setStyleSheet("QTextEdit { background-color: #2d2d2d; color: #d4d4d4; }")
        self.message_input.textChanged.connect(lambda: self.estimate_tokens(self.message_input.toPlainText()))
        input_layout.addWidget(self.message_input)
        
        # Add token counter
        token_layout = QHBoxLayout()
        self.token_label = QLabel("Estimated Tokens: 0 / 4096")
        self.token_label.setStyleSheet("QLabel { color: #d4d4d4; }")
        token_layout.addWidget(self.token_label)
        token_layout.addStretch()
        
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        input_layout.addWidget(send_button)
        
        layout.addLayout(input_layout)
        layout.addLayout(token_layout)
        
        # Add controls
        controls_layout = QHBoxLayout()
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("QComboBox { background-color: #2d2d2d; color: #d4d4d4; }")
        controls_layout.addWidget(QLabel("Model:"))
        controls_layout.addWidget(self.model_combo)
        
        # Temperature control
        controls_layout.addWidget(QLabel("Temperature:"))
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setValue(70)
        self.temperature_slider.valueChanged.connect(self.update_temperature)
        controls_layout.addWidget(self.temperature_slider)
        
        # Preset selection
        controls_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(MODEL_PRESETS.keys())
        self.preset_combo.setCurrentText(self.preset_name)
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        controls_layout.addWidget(self.preset_combo)
        
        layout.addLayout(controls_layout)
        
        # Add status bar
        self.statusBar().addPermanentWidget(QLabel())  # Git sync status
        self.update_git_status()  # Initial status update
        
        # Status update timer
        self.git_status_timer = QTimer(self)
        self.git_status_timer.timeout.connect(self.update_git_status)
        self.git_status_timer.start(60000)  # Update every minute
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
    def load_preset(self, preset_name):
        preset = MODEL_PRESETS[preset_name]
        self.temperature = preset["temperature"]
        self.top_p = preset["top_p"]
        self.top_k = preset["top_k"]
        
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
        estimated_tokens = len(text) // 4
        if hasattr(self, 'token_label'):
            self.token_label.setText(f"Estimated Tokens: {estimated_tokens} / 4096")
        return estimated_tokens

    def set_status(self, message, timeout=3000):
        self.statusBar().showMessage(message)
        self.status_timer.start(timeout)

    def clear_status(self):
        self.statusBar().showMessage("Ready")

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
        self.temperature = self.temperature_slider.value() / 100
        self.temp_value.setText(f"{self.temperature:.2f}")

    def send_message(self):
        """Send a message to the Ollama server."""
        if self.worker and self.worker.isRunning():
            return  # Don't send if already processing
            
        message = self.message_input.toPlainText().strip()
        if not message:
            return

        # Check token limit
        if self.estimate_tokens(message) > 4096:
            self.set_status("Message too long! Please reduce the length.")
            return
            
        # Disable input while processing
        self.message_input.setEnabled(False)
        
        try:
            # Clear input and add user message to display
            self.message_input.clear()
            self.chat_history.append({"role": "user", "content": message})
            self.update_chat_display()
            
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Clean up previous worker if it exists
            if self.worker:
                self.worker.stop()
                self.worker.wait()
                self.worker.deleteLater()
            
            # Create and start worker thread
            self.worker = ChatWorker(
                self.current_model,
                message,
                self.chat_history[:-1],  # Exclude the message we just added
                self.base_url,
                self.system_prompt
            )
            
            # Connect signals
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.chunk_received.connect(self.handle_chunk)
            self.worker.response_received.connect(self.handle_response)
            self.worker.error_occurred.connect(self.handle_error)
            self.worker.progress_updated.connect(self.progress_bar.setValue)
            
            # Start the worker
            self.worker.start()
            
        except Exception as e:
            self.handle_error(f"Failed to send message: {str(e)}")
            self.message_input.setEnabled(True)

    def on_worker_finished(self):
        """Handle worker thread completion."""
        self.message_input.setEnabled(True)
        self.progress_bar.setVisible(False)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def handle_chunk(self, chunk):
        """Handle incoming message chunks with proper formatting."""
        if not hasattr(self, 'current_response'):
            self.current_response = ""
        self.current_response += chunk
        
        # Update the last message in chat display
        if self.chat_history and self.chat_history[-1]["role"] == "assistant":
            self.chat_history[-1]["content"] = self.current_response
        else:
            self.chat_history.append({"role": "assistant", "content": self.current_response})
        
        self.update_chat_display()

    def handle_response(self, response):
        """Handle completed response."""
        self.current_response = ""
        self.save_chat()  # Auto-save after each response

    def handle_error(self, error_message):
        """Handle error messages with proper formatting."""
        self.progress_bar.setVisible(False)
        formatted_error = MessageStyle.format_error(error_message)
        self.chat_display.append(formatted_error)
        self.set_status(f"Error: {error_message}")

    def update_chat_display(self):
        """Update the chat display with formatted messages."""
        self.chat_display.clear()
        formatted_chat = ""
        
        for message in self.chat_history:
            is_user = message.get('role') == 'user'
            content = message.get('content', '')
            formatted_chat += MessageStyle.format_message(content, is_user)
        
        self.chat_display.setHtml(formatted_chat)
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_server_status(self, status):
        """Update server status and UI indicators."""
        self.server_status = status
        status_color = {
            ServerStatus.CONNECTED: "#4CAF50",
            ServerStatus.DISCONNECTED: "#F44336",
            ServerStatus.CONNECTING: "#FFC107",
            ServerStatus.ERROR: "#F44336"
        }.get(status, "#F44336")
        
        self.set_status(f"Server Status: {status}")

    def clear_chat(self):
        """Clear the chat history and display."""
        self.chat_history = []
        self.chat_display.clear()
        self.set_status("Chat cleared")
        
    def load_chat(self):
        """Load a previous chat session."""
        files = [f for f in os.listdir(self.chat_log_dir) if f.endswith('.json')]
        if not files:
            self.set_status("No chat logs found")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Load Chat")
        dialog.setModal(True)
        layout = QVBoxLayout()
        
        list_widget = QListWidget()
        list_widget.addItems(sorted(files, reverse=True))
        layout.addWidget(list_widget)
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addWidget(button_box)
        
        def load_selected():
            selected = list_widget.currentItem()
            if selected:
                try:
                    filepath = os.path.join(self.chat_log_dir, selected.text())
                    with open(filepath, 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)
                        
                    self.chat_history = chat_data.get('messages', [])
                    self.current_model = chat_data.get('model', self.current_model)
                    
                    # Update model selection if available
                    index = self.model_combo.findText(self.current_model)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                    
                    self.update_chat_display()
                    self.set_status(f"Loaded chat from {selected.text()}")
                    dialog.accept()
                except Exception as e:
                    self.set_status(f"Error loading chat: {str(e)}")
        
        button_box.accepted.connect(load_selected)
        button_box.rejected.connect(dialog.reject)
        
        dialog.setLayout(layout)
        dialog.resize(400, 300)
        dialog.exec_()

    def save_chat(self):
        """Save current chat history to a file."""
        if not self.chat_history:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.json"
        filepath = os.path.join(self.chat_log_dir, filename)
        
        chat_data = {
            "model": self.current_model,
            "timestamp": timestamp,
            "messages": self.chat_history
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False)
            self.set_status(f"Chat saved to {filename}")
        except Exception as e:
            self.set_status(f"Error saving chat: {str(e)}")

    def edit_user_name(self):
        name, ok = QInputDialog.getText(self, 'Edit User Name', 'Enter your name:', text=self.user_name)
        if ok and name:
            self.user_name = name
            self.setWindowTitle(f"Ollama Chat - {self.user_name}")
            self.system_prompt = f"You are a helpful AI assistant. The user's name is {self.user_name}."
            self.save_settings()

    def edit_system_prompt(self):
        prompt, ok = QInputDialog.getText(self, 'Edit System Prompt', 'Enter system prompt:', 
                                        text=self.system_prompt)
        if ok:
            self.system_prompt = prompt
            self.save_settings()

    def toggle_git_sync(self):
        if self.git_sync.running:
            self.git_sync.stop()
            self.set_status("Git auto-sync disabled")
        else:
            self.git_sync.start()
            self.set_status("Git auto-sync enabled")
        self.update_git_status()

    def set_git_sync_interval(self):
        current_interval = self.git_sync.interval // 3600  # Convert seconds to hours
        ok, interval = QInputDialog.getInt(
            self,
            'Set Git Sync Interval',
            'Enter sync interval in hours:',
            value=current_interval,
            min=1, max=1000
        )
        if ok:
            self.git_sync.interval = interval * 3600  # Convert hours to seconds
            self.set_status(f"Git sync interval set to {interval} hours")

    def update_git_status(self):
        status = self.git_sync.get_status()
        self.statusBar().findChild(QLabel).setText(status)

    def customize_colors(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Customize Colors")
        layout = QVBoxLayout(dialog)
        
        # Color pickers for different elements
        colors = {
            "Background": "#1e1e1e" if self.dark_mode else "#ffffff",
            "Text": "#d4d4d4" if self.dark_mode else "#000000",
            "User Message": MessageStyle.USER_COLOR,
            "AI Message": MessageStyle.AI_COLOR,
            "Code Background": MessageStyle.CODE_BG,
            "Error": MessageStyle.ERROR_COLOR
        }
        
        color_pickers = {}
        for name, default_color in colors.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"))
            button = QPushButton()
            button.setStyleSheet(f"background-color: {default_color};")
            button.clicked.connect(lambda checked, n=name: self.pick_color(n, button, color_pickers))
            row.addWidget(button)
            layout.addLayout(row)
            color_pickers[name] = button
        
        buttons = QHBoxLayout()
        save = QPushButton("Save")
        save.clicked.connect(lambda: self.save_colors(color_pickers, dialog))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(dialog.reject)
        buttons.addWidget(save)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        
        dialog.exec_()

    def pick_color(self, name, button, color_pickers):
        color = QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()};")
            color_pickers[name].color = color.name()

    def save_colors(self, color_pickers, dialog):
        try:
            colors = {name: button.color for name, button in color_pickers.items()}
            MessageStyle.USER_COLOR = colors.get("User Message", MessageStyle.USER_COLOR)
            MessageStyle.AI_COLOR = colors.get("AI Message", MessageStyle.AI_COLOR)
            MessageStyle.CODE_BG = colors.get("Code Background", MessageStyle.CODE_BG)
            MessageStyle.ERROR_COLOR = colors.get("Error", MessageStyle.ERROR_COLOR)
            
            self.apply_theme(
                bg_color=colors.get("Background"),
                text_color=colors.get("Text")
            )
            
            self.save_settings()
            self.update_chat_display()
            dialog.accept()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save colors: {str(e)}")

    def change_persona(self, persona):
        self.system_prompt = CHAT_PERSONAS[persona]
        self.save_settings()
        self.set_status(f"Changed persona to: {persona}")

    def show_about(self):
        about_text = f"""
        <h2>Ollama Chat</h2>
        <p>Version 1.0.0</p>
        <p>Created by: {self.user_name}</p>
        <p>A modern chat interface for Ollama AI models.</p>
        <p>Current Model: {self.current_model}</p>
        <p> 2024 All rights reserved.</p>
        """
        QMessageBox.about(self, "About Ollama Chat", about_text)

    def show_shortcuts(self):
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        <table>
        <tr><td><b>Ctrl+N:</b></td><td>New Chat</td></tr>
        <tr><td><b>Ctrl+S:</b></td><td>Save Chat</td></tr>
        <tr><td><b>Ctrl+O:</b></td><td>Load Chat</td></tr>
        <tr><td><b>Ctrl+E:</b></td><td>Export Chat</td></tr>
        <tr><td><b>Ctrl+Q:</b></td><td>Exit</td></tr>
        <tr><td><b>Ctrl+Enter:</b></td><td>Send Message</td></tr>
        </table>
        """
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)

    def select_model(self):
        models = self.get_available_models()
        if not models:
            QMessageBox.warning(self, "Error", "Failed to fetch available models. Please ensure Ollama is running.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Model")
        layout = QVBoxLayout(dialog)

        # Model selection
        model_group = QGroupBox("Available Models")
        model_layout = QVBoxLayout()
        
        model_list = QListWidget()
        for model in models:
            item = QListWidgetItem(model)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if model == self.current_model else Qt.Unchecked)
            model_list.addItem(item)
        
        model_layout.addWidget(model_list)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Preset selection
        preset_group = QGroupBox("Model Preset")
        preset_layout = QVBoxLayout()
        
        preset_combo = QComboBox()
        preset_combo.addItems(MODEL_PRESETS.keys())
        preset_combo.setCurrentText("Balanced")  # Default preset
        preset_layout.addWidget(preset_combo)
        
        # Parameter display
        param_layout = QFormLayout()
        param_widgets = {}
        
        for param, value in MODEL_PRESETS["Balanced"].items():
            param_label = QLabel(param.replace("_", " ").title() + ":")
            param_spin = QDoubleSpinBox()
            param_spin.setRange(0.0, 2.0)
            param_spin.setSingleStep(0.1)
            param_spin.setValue(value)
            param_widgets[param] = param_spin
            param_layout.addRow(param_label, param_spin)
        
        preset_layout.addLayout(param_layout)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Update parameters when preset changes
        def update_params():
            preset = preset_combo.currentText()
            preset_values = MODEL_PRESETS[preset]
            for param, value in preset_values.items():
                param_widgets[param].setValue(value)
        
        preset_combo.currentTextChanged.connect(update_params)

        # Buttons
        buttons = QHBoxLayout()
        save = QPushButton("Save")
        cancel = QPushButton("Cancel")
        
        def save_model_settings():
            # Get selected model
            for i in range(model_list.count()):
                item = model_list.item(i)
                if item.checkState() == Qt.Checked:
                    self.current_model = item.text()
                    break
            
            # Save parameters
            preset = preset_combo.currentText()
            if preset == "Custom":
                # Save custom parameters
                custom_params = {}
                for param, widget in param_widgets.items():
                    custom_params[param] = widget.value()
                MODEL_PRESETS["Custom"] = custom_params
            
            self.model_preset = preset
            self.save_settings()
            self.set_status(f"Model changed to: {self.current_model} ({preset} preset)")
            dialog.accept()
        
        save.clicked.connect(save_model_settings)
        cancel.clicked.connect(dialog.reject)
        
        buttons.addWidget(save)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        
        dialog.exec_()

    def edit_model_params(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Model Parameters")
        layout = QVBoxLayout(dialog)
        
        # Parameter inputs
        param_layout = QFormLayout()
        param_widgets = {}
        
        current_preset = MODEL_PRESETS[self.model_preset]
        for param, value in current_preset.items():
            param_label = QLabel(param.replace("_", " ").title() + ":")
            param_spin = QDoubleSpinBox()
            param_spin.setRange(0.0, 2.0)
            param_spin.setSingleStep(0.1)
            param_spin.setValue(value)
            param_widgets[param] = param_spin
            param_layout.addRow(param_label, param_spin)
        
        layout.addLayout(param_layout)
        
        # Buttons
        buttons = QHBoxLayout()
        save = QPushButton("Save as Custom")
        reset = QPushButton("Reset to Default")
        cancel = QPushButton("Cancel")
        
        def save_custom_params():
            custom_params = {}
            for param, widget in param_widgets.items():
                custom_params[param] = widget.value()
            MODEL_PRESETS["Custom"] = custom_params
            self.model_preset = "Custom"
            self.save_settings()
            self.set_status("Saved custom model parameters")
            dialog.accept()
        
        def reset_params():
            self.model_preset = "Balanced"
            self.save_settings()
            self.set_status("Reset to default model parameters")
            dialog.accept()
        
        save.clicked.connect(save_custom_params)
        reset.clicked.connect(reset_params)
        cancel.clicked.connect(dialog.reject)
        
        buttons.addWidget(save)
        buttons.addWidget(reset)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        
        dialog.exec_()

    def get_available_models(self):
        """Fetch available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json()['models']]
                return sorted(models)
        except Exception as e:
            self.set_status(f"Error fetching models: {str(e)}")
        return []

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide style
    app.setStyle('Fusion')
    
    # Create dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    
    # Apply the palette
    app.setPalette(palette)
    
    # Create and show the main window
    window = ChatWindow()
    window.show()
    
    # Start the event loop
    return app.exec_()

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Application error: {str(e)}")
        sys.exit(1)
