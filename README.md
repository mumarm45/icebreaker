# Icebreaker Bot

An AI-powered assistant that generates personalized icebreakers and conversation starters based on LinkedIn profiles.
```
## This project I created to learn RAG with llama-index
```
## Features

- **LinkedIn Profile Analysis**: Extract and analyze LinkedIn profile data
- **AI-Powered Insights**: Generate interesting facts and conversation starters using Claude AI
- **Interactive Chat**: Ask follow-up questions about the profile
- **Web Interface**: User-friendly Gradio interface for easy interaction
- **Mock Data Support**: Test without API keys using sample profile data

## Requirements

- Python 3.12
- Anthropic API key (for Claude LLM)
- ProxyCurl API key (optional, for live LinkedIn data)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/icebreaker.git
   cd icebreaker
   ```

2. Create a virtual environment and install dependencies using `uv`:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```
   
   Or with standard pip but you have to copy and paste the packages in requirements.txt: 
   ```bash
   create requirements.txt and copy the following pacakges in
   _____________________________
   anthropic>=0.75.0,
   gradio>=6.2.0,
   llama-index>=0.14.12,<0.15,
   llama-index-llms-anthropic>=0.10.4,
   llama-index-readers-web>=0.5.6,
   llama-index-embeddings-voyageai>=0.3.0,
   pydantic>=2.12.5,
   requests>=2.32.5,
   _____________________________
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

4. Create a `.env` file with your API keys:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   PROXYCURL_API_KEY=your_proxycurl_api_key  # Optional
   ```

## Usage

### Web Interface (Gradio)

```bash
python app.py
```

This launches a web interface at `http://127.0.0.1:5000` with:
- **Process Profile** tab: Enter a LinkedIn URL and process the profile
- **Chat** tab: Ask questions about the processed profile

### Command Line

```bash
# Using mock data (no API keys required except Anthropic)
python main.py --mock

# Using live LinkedIn data
python main.py
```

## Configuration

Key settings in `config.py`:
- `EMBEDDING_PROVIDER`: Choose between `huggingface`, `openai`, or `voyage`
- `LLM_PROVIDER`: Currently supports `anthropic`
- `LLM_MODEL_ID`: Claude model to use (default: `claude-3-5-haiku-20241022`)

## Project Structure

```
icebreaker/
├── app.py                 # Gradio web interface
├── main.py                # CLI entry point
├── config.py              # Configuration and environment variables
└── src/modules/
    ├── data_extraction.py # LinkedIn profile extraction
    ├── data_processing.py # Profile data processing and vector DB
    └── llm_interface.py   # LLM and embedding model setup
```



## License

MIT
