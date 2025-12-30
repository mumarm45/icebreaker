# Icebreaker Bot

An AI-powered assistant that generates personalized icebreakers and conversation starters based on LinkedIn profiles.

## Features

- **LinkedIn Profile Analysis**: Extract and analyze LinkedIn profile data
- **AI-Powered Insights**: Generate interesting facts and conversation starters using Claude AI
- **Interactive Chat**: Ask follow-up questions about the profile
- **Web Interface**: User-friendly Gradio interface for easy interaction
- **Mock Data Support**: Test without API keys using sample profile data

## Requirements

- Python 3.11 - 3.13
- Anthropic API key (for Claude LLM)
- ProxyCurl API key (optional, for live LinkedIn data)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/icebreaker.git
   cd icebreaker
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

3. Create a `.env` file with your API keys:
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
