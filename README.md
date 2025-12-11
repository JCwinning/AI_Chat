# 🤖 AI Chat Application

A multilingual, multi-model AI chat application built with Streamlit that supports streaming responses, image inputs, and concurrent model querying.

## ✨ Features

- **🌐 Multilingual Support**: English and Chinese interface
- **🤖 Multi-Model Chat**: Query multiple AI models simultaneously with side-by-side comparison
- **🎨 Image Generation**: Create and edit images using advanced AI models
- **📚 Prompt Bay**: Library of system prompts for specialized AI interactions
- **📱 Responsive Design**: Clean, modern two-panel layout with sidebar configuration
- **🖼️ File Support**: Upload images and PDF files for multimodal AI analysis
- **🔍 Web Search**: Integrated web search capabilities for real-time information
- **💬 Chat History**: Persistent conversation storage with automatic save
- **⚡ Streaming Responses**: Real-time response display from multiple models
- **🔧 Custom Models**: Add your own AI model configurations dynamically
- **🌍 Multi-Provider**: Support for OpenAI, ModelScope, OpenRouter, and more

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository** (if applicable) or download the files:
```bash
# If using git
git clone <repository-url>
cd AI_Chat

# Otherwise, ensure you have all the project files
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure models**:
Create a `config.py` file with your AI model configurations (see Configuration section below).

4. **Run the application**:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## ⚙️ Configuration

### Setting up `config.py`

Create a `config.py` file in the project root with your model configurations. Use the following format:

```python

# OpenAI Configuration
openai_gpt4 = {
    "api_key": "your-openai-api-key-here",
    "base_url": "https://api.openai.com/v1",
    "models": "gpt-4",
}


# OpenRouter Configuration
openrouter_mistral = {
    "api_key": "your-openrouter-api-key-here",
    "base_url": "https://openrouter.ai/api/v1",
    "models": "mistralai/mistral-7b-instruct",
}

# Custom Model Example
custom_llm = {
    "api_key": "your-api-key",
    "base_url": "https://your-custom-endpoint.com/v1",
    "models": "your-model-name",
}

# Image Generation Model Example
gemini_image = {
    "api_key": "your-openrouter-key",
    "base_url": "https://openrouter.ai/api/v1",
    "models": "google/gemini-2.5-flash-image-preview",
    "type": "image"  # Specify type for image generation models
}
```


## 📖 Usage Guide

### Basic Chat

1. **Select Models**: Choose one or more AI models from the sidebar
2. **Configure Parameters**: Adjust temperature and system prompt as needed
3. **Start Chatting**: Type your message and press Enter
4. **View Responses**: See streaming responses from all selected models side-by-side

### Advanced Features

#### File Upload
- Click "📎 Upload Files" in the sidebar
- Select images (PNG, JPG, JPEG) or PDF files
- Files are displayed before sending
- Include images in your message for multimodal AI analysis
- PDF files are automatically converted to Markdown and sent to AI
- Support for Word documents (.docx) and Excel files (.xlsx)

#### Image Generation
- Select an image generation model (e.g., Gemini Flash Image, Qwen Image)
- Describe the image you want to create
- **Multimodal Generation**: Upload an image and ask the model to "modify this" or "generate a variation"

#### Chat Management
- **New Chat**: Click "➕ New Chat" to start a fresh conversation
- **Saved Chats**: Access previous conversations from the sidebar
- **Delete Chats**: Remove unwanted conversations with the 🗑️ button
- **Auto-save**: Chats are automatically saved with timestamps

#### Custom Models
1. Click "➕ Add New Model" in the Models tab
2. Fill in the model details:
   - **Model Name**: Display name for the model
   - **API Key**: Your API credentials
   - **Base URL**: API endpoint URL
   - **Model(s)**: Comma-separated model names
3. Click "Add Model" to save and automatically select it

#### Prompt Bay
1. Navigate to the "📚 Prompt Bay" tab
2. Browse through a library of pre-configured system prompts
3. Use the search bar to find specific prompts
4. Click "Apply" to use a prompt as your system prompt
5. Prompts include roles like "Linux Terminal", "Travel Guide", "Password Generator", and more

#### Web Search
1. Click the toggle switch in the chat interface to enable web search
2. Your message will be processed with real-time web information
3. Search results are displayed with sources and dates
4. AI models will incorporate up-to-date information in their responses

#### Language Switching
- Toggle between English and Chinese using the language button in the sidebar
- All interface elements will update to your preferred language

## 🏗️ Architecture

### Project Structure

```
AI_Chat/
├── app.py                 # Main Streamlit application
├── config.py              # Model configurations (user-created)
├── requirements.txt       # Python dependencies
├── prompt_bay.csv        # System prompts library
├── design.md             # Detailed technical documentation
├── chat_history.json     # Persistent chat storage (auto-generated)
├── CLAUDE.md             # Development guide for AI assistants
├── README.md             # This file
└── README_CN.md          # Chinese version
```

### Key Components

- **Multi-threading**: Parallel model requests using Python threading
- **Queue Communication**: Thread-safe message passing for streaming responses
- **Session Management**: Streamlit session state for chat history and settings
- **File Persistence**: JSON-based chat history storage
- **File Processing**: Base64 encoding for images and Markdown conversion for PDFs
- **Error Handling**: Graceful degradation for API failures

## 🛠️ Development

### Dependencies

The application uses the following main dependencies:
- `streamlit>=1.28.0` - Web framework
- `pandas>=2.0.0` - Data processing for CSV files (Prompt Bay)
- `openai>=1.0.0` - API client library
- `anthropic>=0.7.0` - Anthropic API support
- `pillow>=10.0.0` - Image processing
- `markitdown>=0.0.1a2` - PDF to Markdown conversion
- `python-dotenv>=1.0.0` - Environment variable management
- `keyring` - Secure credential storage
- `dashscope>=1.14.0` - Alibaba Cloud AI model support
- `openpyxl>=3.0.0` - Excel file processing
- `python-docx>=1.2.0` - Word document processing
- `tavily-python>=0.3.0` - Web search integration

### Adding New Features

1. **New Models**: Add configurations to `config.py`
2. **UI Changes**: Modify `app.py` while maintaining the multilingual structure
3. **New Languages**: Extend the `translations` dictionary in `app.py`
4. **Storage Formats**: Modify the save/load functions for different persistence

## 🔒 Security

- **API Keys**: Store securely using keyring or environment variables
- **File Uploads**: Support for images (PNG, JPG, JPEG) and PDF files
- **Session Data**: Chat history stored locally in JSON format
- **No Remote Storage**: All data remains on your local machine

