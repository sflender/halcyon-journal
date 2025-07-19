#!/usr/bin/env python3
"""
Halcyon Journal - GUI Version
A beautiful, native macOS journaling app with local AI processing.
"""

import sys
import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QTabWidget, QScrollArea,
    QFrame, QSplitter, QMessageBox, QProgressBar, QComboBox,
    QFileDialog, QMenuBar, QStatusBar, QDialog, QLineEdit
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPalette, QColor, QIcon, QAction
import markdown

class AIWorker(QThread):
    """Background worker for AI processing."""
    response_ready = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, prompt: str, model_name: str):
        super().__init__()
        self.prompt = prompt
        self.model_name = model_name
    
    def run(self):
        try:
            client = ollama.Client()
            response = client.generate(
                model=self.model_name,
                prompt=self.prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 400
                }
            )
            self.response_ready.emit(response['response'])
        except Exception as e:
            self.error_occurred.emit(str(e))

class JournalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.journal_dir = Path.home() / ".halcyon_journal"
        self.journal_dir.mkdir(exist_ok=True)
        self.embeddings_file = self.journal_dir / "embeddings.json"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ollama_client = ollama.Client()
        self.model_name = 'llama2'
        
        self.init_ui()
        self.setup_ollama()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("🌟 Halcyon Journal")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set up the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create main layout
        layout = QHBoxLayout(main_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_journal_tab()
        self.create_reflect_tab()
        self.create_settings_tab()
        
        # Set up menu bar
        self.create_menu_bar()
        
        # Set up status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Apply macOS-style styling
        self.apply_styling()
        
    def create_journal_tab(self):
        """Create the journaling tab."""
        journal_widget = QWidget()
        layout = QVBoxLayout(journal_widget)
        
        # Header
        header = QLabel("📝 Daily Journal")
        header.setFont(QFont("SF Pro Display", 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Date display
        self.date_label = QLabel(f"Today: {datetime.date.today().strftime('%B %d, %Y')}")
        self.date_label.setFont(QFont("SF Pro Display", 14))
        self.date_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.date_label)
        
        # Journal entry area
        entry_label = QLabel("Write about your day:")
        entry_label.setFont(QFont("SF Pro Display", 12, QFont.Weight.Medium))
        layout.addWidget(entry_label)
        
        self.journal_text = QTextEdit()
        self.journal_text.setFont(QFont("SF Pro Display", 12))
        self.journal_text.setPlaceholderText("Start writing about your day...\n\nPress Ctrl+Enter to save and get AI reflection.")
        self.journal_text.setMinimumHeight(200)
        layout.addWidget(self.journal_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("💾 Save Entry")
        self.save_button.clicked.connect(self.save_journal_entry)
        self.save_button.setFont(QFont("SF Pro Display", 12))
        button_layout.addWidget(self.save_button)
        
        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.clicked.connect(self.clear_journal_entry)
        self.clear_button.setFont(QFont("SF Pro Display", 12))
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        
        # AI Response area
        response_label = QLabel("🤗 AI Reflection:")
        response_label.setFont(QFont("SF Pro Display", 12, QFont.Weight.Medium))
        layout.addWidget(response_label)
        
        self.ai_response = QTextEdit()
        self.ai_response.setFont(QFont("SF Pro Display", 11))
        self.ai_response.setReadOnly(True)
        self.ai_response.setMaximumHeight(150)
        self.ai_response.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        layout.addWidget(self.ai_response)
        
        # Progress bar for AI processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Similar entries area
        similar_label = QLabel("📚 Related Past Entries:")
        similar_label.setFont(QFont("SF Pro Display", 12, QFont.Weight.Medium))
        layout.addWidget(similar_label)
        
        self.similar_entries = QTextEdit()
        self.similar_entries.setFont(QFont("SF Pro Display", 10))
        self.similar_entries.setReadOnly(True)
        self.similar_entries.setMaximumHeight(120)
        self.similar_entries.setStyleSheet("background-color: #fff3cd; border: 1px solid #ffeaa7;")
        layout.addWidget(self.similar_entries)
        
        self.tab_widget.addTab(journal_widget, "📝 Journal")
        
    def create_reflect_tab(self):
        """Create the reflection tab."""
        reflect_widget = QWidget()
        layout = QVBoxLayout(reflect_widget)
        
        # Header
        header = QLabel("🔍 Reflect on Your Journal")
        header.setFont(QFont("SF Pro Display", 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Query input
        query_label = QLabel("Ask a question about your journal:")
        query_label.setFont(QFont("SF Pro Display", 12, QFont.Weight.Medium))
        layout.addWidget(query_label)
        
        self.query_input = QLineEdit()
        self.query_input.setFont(QFont("SF Pro Display", 12))
        self.query_input.setPlaceholderText("e.g., 'How have I been feeling lately?' or 'What patterns do you see?'")
        self.query_input.returnPressed.connect(self.ask_reflection_question)
        layout.addWidget(self.query_input)
        
        # Quick question buttons
        quick_questions_label = QLabel("Quick Questions:")
        quick_questions_label.setFont(QFont("SF Pro Display", 11, QFont.Weight.Medium))
        layout.addWidget(quick_questions_label)
        
        quick_buttons_layout = QHBoxLayout()
        
        questions = [
            "How have I been doing?",
            "What patterns do you see?",
            "When was I most productive?",
            "What challenges have I faced?"
        ]
        
        for question in questions:
            btn = self.create_question_button(question)
            quick_buttons_layout.addWidget(btn)
        
        layout.addLayout(quick_buttons_layout)
        
        # Ask button
        self.ask_button = QPushButton("🤔 Ask Question")
        self.ask_button.clicked.connect(self.ask_reflection_question)
        self.ask_button.setFont(QFont("SF Pro Display", 12))
        layout.addWidget(self.ask_button)
        
        # Response area
        response_label = QLabel("💭 AI Reflection:")
        response_label.setFont(QFont("SF Pro Display", 12, QFont.Weight.Medium))
        layout.addWidget(response_label)
        
        self.reflect_response = QTextEdit()
        self.reflect_response.setFont(QFont("SF Pro Display", 11))
        self.reflect_response.setReadOnly(True)
        self.reflect_response.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        layout.addWidget(self.reflect_response)
        
        self.tab_widget.addTab(reflect_widget, "🔍 Reflect")
        
    def create_settings_tab(self):
        """Create the settings tab."""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        # Header
        header = QLabel("⚙️ Settings")
        header.setFont(QFont("SF Pro Display", 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # AI Model selection
        model_label = QLabel("AI Model:")
        model_label.setFont(QFont("SF Pro Display", 12, QFont.Weight.Medium))
        layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(['llama2', 'mistral', 'llama2:7b', 'llama2:3b'])
        self.model_combo.setCurrentText(self.model_name)
        self.model_combo.currentTextChanged.connect(self.change_model)
        layout.addWidget(self.model_combo)
        
        # Journal directory
        dir_label = QLabel("Journal Directory:")
        dir_label.setFont(QFont("SF Pro Display", 12, QFont.Weight.Medium))
        layout.addWidget(dir_label)
        
        dir_layout = QHBoxLayout()
        self.dir_display = QLineEdit(str(self.journal_dir))
        self.dir_display.setReadOnly(True)
        dir_layout.addWidget(self.dir_display)
        
        change_dir_btn = QPushButton("Change")
        change_dir_btn.clicked.connect(self.change_journal_directory)
        dir_layout.addWidget(change_dir_btn)
        layout.addLayout(dir_layout)
        
        # Stats
        stats_label = QLabel("📊 Journal Statistics:")
        stats_label.setFont(QFont("SF Pro Display", 12, QFont.Weight.Medium))
        layout.addWidget(stats_label)
        
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMaximumHeight(150)
        self.update_stats()
        layout.addWidget(self.stats_display)
        
        # Export/Import/Backup buttons
        export_layout = QHBoxLayout()
        
        backup_btn = QPushButton("💾 Backup Journal")
        backup_btn.clicked.connect(self.backup_journal)
        export_layout.addWidget(backup_btn)
        
        export_btn = QPushButton("📤 Export Journal")
        export_btn.clicked.connect(self.export_journal)
        export_layout.addWidget(export_btn)
        
        import_btn = QPushButton("📥 Import Journal")
        import_btn.clicked.connect(self.import_journal)
        export_layout.addWidget(import_btn)
        
        layout.addLayout(export_layout)
        
        layout.addStretch()
        
        self.tab_widget.addTab(settings_widget, "⚙️ Settings")
        
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_entry_action = QAction('New Entry', self)
        new_entry_action.setShortcut('Ctrl+N')
        new_entry_action.triggered.connect(self.new_journal_entry)
        file_menu.addAction(new_entry_action)
        
        save_action = QAction('Save Entry', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_journal_entry)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def apply_styling(self):
        """Apply macOS-style styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #c0c0c0;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom: 1px solid #ffffff;
            }
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0056CC;
            }
            QPushButton:pressed {
                background-color: #004499;
            }
            QTextEdit {
                border: 1px solid #c0c0c0;
                border-radius: 6px;
                padding: 8px;
                background-color: #ffffff;
            }
            QLineEdit {
                border: 1px solid #c0c0c0;
                border-radius: 6px;
                padding: 8px;
                background-color: #ffffff;
            }
            QComboBox {
                border: 1px solid #c0c0c0;
                border-radius: 6px;
                padding: 8px;
                background-color: #ffffff;
            }
        """)
        
    def setup_ollama(self):
        """Setup Ollama and check for models."""
        try:
            models = self.ollama_client.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.model_name not in model_names:
                self.status_bar.showMessage("Downloading AI model... This may take a few minutes.")
                self.ollama_client.pull(self.model_name)
                self.status_bar.showMessage("AI model ready!")
            else:
                self.status_bar.showMessage("AI model ready!")
                
        except Exception as e:
            QMessageBox.warning(self, "Ollama Error", 
                              f"Error setting up Ollama: {str(e)}\n\nPlease ensure Ollama is installed and running.")
    
    def save_journal_entry(self):
        """Save the current journal entry."""
        content = self.journal_text.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, "Empty Entry", "Please write something before saving.")
            return
        
        # Save to file
        today_file = self.journal_dir / self._get_today_filename()
        with open(today_file, 'a') as f:
            if today_file.exists() and os.path.getsize(today_file) > 0:
                f.write("\n\n---\n\n")
            f.write(content)
        
        # Update embeddings
        self._update_embeddings(content)
        
        # Get AI response
        self.get_ai_response(content)
        
        # Find similar entries
        similar_entries = self._find_similar_entries(content)
        self.display_similar_entries(similar_entries)
        
        self.status_bar.showMessage(f"Entry saved to {today_file}")
        
    def get_ai_response(self, content: str):
        """Get AI response for the journal entry."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Find similar entries for context
        similar_entries = self._find_similar_entries(content)
        
        # Build prompt
        context = ""
        if similar_entries:
            context = "\n\nRelated past entries:\n"
            for entry in similar_entries[:2]:  # Limit context
                context += f"\n--- {entry['filename']} ---\n{entry['content'][:300]}...\n"
        
        prompt = f"""You are a caring, empathetic therapist. The user has written a journal entry about their day. 
        Please provide a thoughtful, supportive response that:
        1. Acknowledges their feelings and experiences
        2. Offers gentle insights or observations
        3. Asks a thoughtful follow-up question
        4. Maintains a warm, professional tone
        
        Current journal entry:
        {content}
        
        {context}
        
        Your response:"""
        
        # Start AI worker
        self.ai_worker = AIWorker(prompt, self.model_name)
        self.ai_worker.response_ready.connect(self.handle_ai_response)
        self.ai_worker.error_occurred.connect(self.handle_ai_error)
        self.ai_worker.start()
        
    def handle_ai_response(self, response: str):
        """Handle AI response."""
        self.progress_bar.setVisible(False)
        self.ai_response.setPlainText(response)
        
    def handle_ai_error(self, error: str):
        """Handle AI error."""
        self.progress_bar.setVisible(False)
        self.ai_response.setPlainText("I'm here to listen and support you. Your journal entry shows thoughtful reflection. How are you feeling about what you've written today?")
        self.status_bar.showMessage(f"AI Error: {error}")
        
    def create_question_button(self, question):
        """Create a button for a specific question."""
        btn = QPushButton(question)
        btn.setFont(QFont("SF Pro Display", 10))
        btn.clicked.connect(lambda: self.ask_specific_question(question))
        return btn
        
    def ask_reflection_question(self):
        """Ask a reflection question."""
        query = self.query_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Empty Query", "Please enter a question.")
            return
        
        self.ask_specific_question(query)
        
    def ask_specific_question(self, question: str):
        """Ask a specific question."""
        self.query_input.setText(question)
        self.reflect_response.setPlainText("Thinking...")
        
        # Find relevant entries
        similar_entries = self._find_similar_entries(question, top_k=5)
        
        if not similar_entries:
            self.reflect_response.setPlainText("No relevant journal entries found. Try writing some entries first!")
            return
        
        # Build context
        context = "\n\n".join([
            f"Entry from {entry['filename']}:\n{entry['content']}"
            for entry in similar_entries
        ])
        
        prompt = f"""You are a helpful assistant analyzing someone's personal journal. 
        The user asked: "{question}"
        
        Based on these journal entries, provide a thoughtful, insightful response:
        
        {context}
        
        Your response:"""
        
        # Start AI worker
        self.reflect_worker = AIWorker(prompt, self.model_name)
        self.reflect_worker.response_ready.connect(self.handle_reflect_response)
        self.reflect_worker.error_occurred.connect(self.handle_reflect_error)
        self.reflect_worker.start()
        
    def handle_reflect_response(self, response: str):
        """Handle reflection response."""
        self.reflect_response.setPlainText(response)
        
    def handle_reflect_error(self, error: str):
        """Handle reflection error."""
        self.reflect_response.setPlainText("I couldn't process your question right now. Please try again later.")
        self.status_bar.showMessage(f"AI Error: {error}")
        
    def _get_today_filename(self) -> str:
        """Get today's journal filename."""
        today = datetime.date.today()
        return f"{today.strftime('%Y-%m-%d')}.md"
        
    def _update_embeddings(self, content: str):
        """Update embeddings for the content."""
        embeddings = self._load_embeddings()
        embeddings[self._get_today_filename()] = self.model.encode(content).tolist()
        self._save_embeddings(embeddings)
        
    def _load_embeddings(self) -> Dict[str, List[float]]:
        """Load existing embeddings."""
        if self.embeddings_file.exists():
            with open(self.embeddings_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_embeddings(self, embeddings: Dict[str, List[float]]):
        """Save embeddings to file."""
        with open(self.embeddings_file, 'w') as f:
            json.dump(embeddings, f)
            
    def _find_similar_entries(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find similar journal entries."""
        embeddings = self._load_embeddings()
        if not embeddings:
            return []
        
        query_embedding = self.model.encode(query)
        
        similarities = []
        for filename, embedding in embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((filename, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        similar_entries = []
        for filename, similarity in similarities[:top_k]:
            entry_path = self.journal_dir / filename
            if entry_path.exists():
                with open(entry_path, 'r') as f:
                    content = f.read()
                    similar_entries.append({
                        'filename': filename,
                        'content': content,
                        'similarity': similarity
                    })
        
        return similar_entries
        
    def display_similar_entries(self, entries: List[Dict]):
        """Display similar entries."""
        if not entries:
            self.similar_entries.setPlainText("No similar entries found.")
            return
        
        text = "Related past entries:\n\n"
        for entry in entries:
            text += f"📅 {entry['filename']} (similarity: {entry['similarity']:.2f})\n"
            text += f"{entry['content'][:150]}...\n\n"
        
        self.similar_entries.setPlainText(text)
        
    def clear_journal_entry(self):
        """Clear the journal entry."""
        self.journal_text.clear()
        
    def new_journal_entry(self):
        """Start a new journal entry."""
        self.tab_widget.setCurrentIndex(0)
        self.journal_text.clear()
        self.ai_response.clear()
        self.similar_entries.clear()
        
    def change_model(self, model_name: str):
        """Change the AI model."""
        self.model_name = model_name
        self.status_bar.showMessage(f"AI model changed to {model_name}")
        
    def change_journal_directory(self):
        """Change the journal directory with data migration."""
        old_dir = self.journal_dir
        
        # Check if current directory has data
        current_entries = list(old_dir.glob("*.md"))
        has_current_data = len(current_entries) > 0
        
        if has_current_data:
            # Ask user what to do with existing data
            reply = QMessageBox.question(
                self, 
                "Existing Journal Data", 
                f"Your current journal directory has {len(current_entries)} entries.\n\n"
                "What would you like to do?\n\n"
                "• 'Yes' = Move all data to the new directory\n"
                "• 'No' = Keep data in current directory (use new directory for future entries)\n"
                "• 'Cancel' = Keep current directory",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                return
            elif reply == QMessageBox.StandardButton.Yes:
                # Move data to new directory
                new_dir = QFileDialog.getExistingDirectory(self, "Select New Journal Directory")
                if not new_dir:
                    return
                    
                new_dir_path = Path(new_dir)
                new_dir_path.mkdir(exist_ok=True)
                
                # Move all journal files
                moved_count = 0
                for file in current_entries:
                    try:
                        import shutil
                        shutil.move(str(file), str(new_dir_path / file.name))
                        moved_count += 1
                    except Exception as e:
                        QMessageBox.warning(self, "Move Error", f"Failed to move {file.name}: {str(e)}")
                
                # Move embeddings file if it exists
                old_embeddings = old_dir / "embeddings.json"
                if old_embeddings.exists():
                    try:
                        import shutil
                        shutil.move(str(old_embeddings), str(new_dir_path / "embeddings.json"))
                    except Exception as e:
                        QMessageBox.warning(self, "Move Error", f"Failed to move embeddings: {str(e)}")
                
                # Update app to use new directory
                self.journal_dir = new_dir_path
                self.embeddings_file = self.journal_dir / "embeddings.json"
                self.dir_display.setText(str(self.journal_dir))
                
                QMessageBox.information(
                    self, 
                    "Directory Changed", 
                    f"Successfully moved {moved_count} journal entries to the new directory."
                )
                
            else:  # No - use new directory for future entries only
                new_dir = QFileDialog.getExistingDirectory(self, "Select New Journal Directory")
                if new_dir:
                    self.journal_dir = Path(new_dir)
                    self.journal_dir.mkdir(exist_ok=True)
                    self.embeddings_file = self.journal_dir / "embeddings.json"
                    self.dir_display.setText(str(self.journal_dir))
                    
                    QMessageBox.information(
                        self, 
                        "Directory Changed", 
                        "New directory set for future entries. Existing entries remain in the old location."
                    )
        else:
            # No existing data, just change directory
            new_dir = QFileDialog.getExistingDirectory(self, "Select Journal Directory")
            if new_dir:
                self.journal_dir = Path(new_dir)
                self.journal_dir.mkdir(exist_ok=True)
                self.embeddings_file = self.journal_dir / "embeddings.json"
                self.dir_display.setText(str(self.journal_dir))
        
        self.update_stats()
            
    def update_stats(self):
        """Update journal statistics."""
        journal_files = list(self.journal_dir.glob("*.md"))
        total_entries = len(journal_files)
        
        if total_entries == 0:
            stats = "No journal entries yet.\nStart writing to see your statistics!"
        else:
            # Calculate some basic stats
            total_words = 0
            for file in journal_files:
                with open(file, 'r') as f:
                    content = f.read()
                    total_words += len(content.split())
            
            avg_words = total_words // total_entries if total_entries > 0 else 0
            
            stats = f"""📊 Journal Statistics:
            
Total Entries: {total_entries}
Total Words: {total_words:,}
Average Words per Entry: {avg_words}
Journal Directory: {self.journal_dir}"""
        
        self.stats_display.setPlainText(stats)
        
    def backup_journal(self):
        """Create a backup of all journal data."""
        backup_dir = QFileDialog.getExistingDirectory(self, "Select Backup Directory")
        if backup_dir:
            backup_path = Path(backup_dir)
            backup_path.mkdir(exist_ok=True)
            
            # Create timestamped backup folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_folder = backup_path / f"halcyon_backup_{timestamp}"
            backup_folder.mkdir(exist_ok=True)
            
            # Copy all journal files
            copied_count = 0
            for file in self.journal_dir.glob("*.md"):
                try:
                    import shutil
                    shutil.copy2(file, backup_folder)
                    copied_count += 1
                except Exception as e:
                    QMessageBox.warning(self, "Backup Error", f"Failed to backup {file.name}: {str(e)}")
            
            # Copy embeddings file
            if self.embeddings_file.exists():
                try:
                    import shutil
                    shutil.copy2(self.embeddings_file, backup_folder)
                except Exception as e:
                    QMessageBox.warning(self, "Backup Error", f"Failed to backup embeddings: {str(e)}")
            
            QMessageBox.information(
                self, 
                "Backup Complete", 
                f"Successfully backed up {copied_count} journal entries to:\n{backup_folder}"
            )
    
    def export_journal(self):
        """Export journal entries."""
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if export_dir:
            # Copy all journal files
            for file in self.journal_dir.glob("*.md"):
                import shutil
                shutil.copy2(file, export_dir)
            QMessageBox.information(self, "Export Complete", f"Journal entries exported to {export_dir}")
            
    def import_journal(self):
        """Import journal entries."""
        import_dir = QFileDialog.getExistingDirectory(self, "Select Import Directory")
        if import_dir:
            # Copy all markdown files
            import_path = Path(import_dir)
            for file in import_path.glob("*.md"):
                import shutil
                shutil.copy2(file, self.journal_dir)
            QMessageBox.information(self, "Import Complete", "Journal entries imported successfully!")
            self.update_stats()
            
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About Halcyon Journal",
                         """🌟 Halcyon Journal v1.0

A privacy-focused, local-only interactive journaling app.

Features:
• Local AI processing with Ollama
• Semantic search for related entries
• Beautiful native macOS interface
• Complete privacy - no cloud processing

Built with PyQt6 and local LLMs.
Your thoughts stay yours! 🌟""")

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Halcyon Journal")
    app.setApplicationVersion("1.0")
    
    # Set application icon (if available)
    # app.setWindowIcon(QIcon("icon.png"))
    
    window = JournalApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 