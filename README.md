# MCP Research Chatbot

An intelligent research assistant that combines the power of Google's Gemini AI with multiple MCP servers including arXiv paper search, web fetching, and filesystem operations through the Model Context Protocol (MCP).

## Features

- **Paper Search**: Search arXiv for academic papers on any topic
- **Web Fetching**: Retrieve and analyze web content via fetch MCP server
- **File Operations**: Read, write, and manage files through filesystem MCP server
- **Smart Storage**: Organizes papers by topic in structured JSON files
- **Interactive Chat**: Natural language interface with Gemini AI
- **Resource Browser**: Browse saved papers by topic
- **Prompt Templates**: Pre-built prompts for research tasks
- **Multi-Server Integration**: Seamless integration of multiple MCP servers

## Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Installation

1. Clone and install dependencies:
```bash
uv add python-dotenv google-genai mcp arxiv
```

2. Create `.env` file:
```
API_KEY=your_gemini_api_key_here
```

3. Create `server_config.json`:
```json
{
  "mcpServers": {
    "research": {
      "command": "python",
      "args": ["server.py"]
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
    }
  }
}
```

### Usage

Start the chatbot:
```bash
python client_chatbot.py
```

## Commands

| Command | Description |
|---------|-------------|
| `search for papers on quantum computing` | Natural language search |
| `fetch content from https://example.com` | Retrieve web content |
| `read file contents from document.txt` | Access filesystem operations |
| `@folders` | List available research topics |
| `@machine_learning` | View papers for specific topic |
| `/prompts` | List available prompt templates |
| `/prompt generate_search_prompt topic=AI` | Execute research prompt |
| `quit` | Exit chatbot |

## Example Workflow

1. **Search**: "Find papers on machine learning optimization"
2. **Browse**: `@folders` to see saved topics
3. **Explore**: `@machine_learning` to view detailed paper info
4. **Analyze**: Use built-in prompts for deeper research synthesis

## Architecture

- **Client** (`client_chatbot.py`): Gemini-powered chat interface with MCP integration
- **Server** (`server.py`): FastMCP server providing arXiv search and paper management
- **Storage**: Papers organized in `papers/` directory by topic

## Tools Available

- `search_papers(topic, max_results)`: Search and save arXiv papers
- `extract_info(paper_id)`: Get detailed info for specific paper
- Resources for browsing topics and papers
- Research prompt templates for comprehensive analysis

## Benefits

- **Persistent Storage**: Papers are saved locally for offline access
- **Structured Organization**: Topics automatically organized in folders
- **AI-Powered Analysis**: Gemini provides intelligent synthesis of research
- **Extensible**: Easy to add new tools and capabilities via MCP

Perfect for researchers, students, and anyone needing to quickly explore and organize academic literature!