# 🌟 Halcyon Journal

A **privacy-focused, local-only interactive journaling app** that processes everything on your device. Your thoughts stay yours.

## ✨ Features

- **📝 Journal Mode**: Write daily entries with intelligent therapist-like responses
- **🔍 Reflect Mode**: Ask questions about your journal and get insights
- **🤖 Local AI**: Uses Ollama with local LLM for all processing
- **🔒 Privacy First**: Everything stays on your device - no cloud processing
- **📚 Smart Search**: Find related entries using semantic similarity
- **📄 Markdown Storage**: Entries saved as plain Markdown files
- **🎨 Beautiful UI**: Rich terminal interface with colors and formatting

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** (required for local AI processing):
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Or visit: https://ollama.ai/download
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
python journal_app.py
```

On first run, the app will:
- Download the Llama2 model (may take a few minutes)
- Create a journal directory at `~/.halcyon_journal`
- Set up semantic search capabilities

## 📖 How to Use

### Journal Mode
1. Select "📝 Journal" from the main menu
2. Write about your day (press Enter twice to finish)
3. Get a thoughtful, therapist-like response
4. Browse related past entries

### Reflect Mode
1. Select "🔍 Reflect" from the main menu
2. Ask questions like:
   - "How have I been doing lately?"
   - "What patterns do you see in my mood?"
   - "When was I most productive?"
   - "What challenges have I been facing?"

## 🗂️ File Structure

```
~/.halcyon_journal/
├── 2024-01-15.md          # Daily journal entries
├── 2024-01-16.md
├── embeddings.json        # Semantic search data
└── ...
```

## 🔧 Technical Details

- **Local LLM**: Uses Ollama with Llama2 model
- **Semantic Search**: Sentence transformers for finding related entries
- **Storage**: Plain Markdown files for maximum portability
- **Privacy**: Zero network requests - everything processes locally

## 🛠️ Customization

### Using Different LLM Models

You can modify the model in `journal_app.py`:

```python
# Change from llama2 to another model
self.model_name = 'mistral'  # or 'codellama', 'llama2:7b', etc.
```

### Journal Directory Location

Change the journal directory by modifying:

```python
self.journal_dir = Path.home() / ".halcyon_journal"  # Change this path
```

## 🔒 Privacy & Security

- **No cloud processing**: All AI responses generated locally
- **No data collection**: Your journal entries never leave your device
- **Plain text storage**: Markdown files are human-readable and portable
- **Local embeddings**: Semantic search data stored locally

## 🐛 Troubleshooting

### Ollama Connection Issues
```bash
# Start Ollama service
ollama serve

# Check available models
ollama list

# Pull a specific model
ollama pull llama2
```

### Python Dependencies
```bash
# If you encounter import errors
pip install --upgrade pip
pip install -r requirements.txt
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License - feel free to use and modify as you wish.

---

**🌟 Happy Journaling!** Your thoughts are precious - keep them private and secure. 