# MCP Integration System

A Chainlit-based chat interface that connects to multiple MCP (Model Control Protocol) servers, allowing users to interact with different tool providers through a unified interface.

## Features

- Connect to multiple MCP servers simultaneously
- Process document uploads with automatic conversion to Markdown
- Store and retrieve documents across different repositories
- Execute tools from various servers through natural language
- View document activity across connected systems

## Prerequisites

- Docker
- Python 3.9+
- OpenAI API key (for GPT-4 integration)
- MongoDB instance (for document storage)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install chainlit openai mcp-client asyncio
```

### 2. Environment Setup

Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Build the Markitdown Docker Image

The system requires a custom Markitdown MCP server image. Build it using:

```bash
# Clone the Markitdown repo if needed
cd mcp

# use https://github.com/microsoft/markitdown/tree/main/packages/markitdown-mcp Markdown MCP to build docker image if you can't access markdown folder.
# Build the image
cd markitdown/
docker build -t markitdown-mcp-server .

cd document-ingestion
docker build -t mcp/document-ingestion .

cd sqlite
docker build -t mcp/sqlite .
```

### 5. MongoDB Setup

Ensure MongoDB is running and accessible at `mongodb://localhost:27017`. You can start MongoDB using Docker:

```bash
docker run -d -p 27017:27017 --name mongodb mongo
```

## Running the Application

Start the application with Chainlit:

```bash
chainlit run chainlit-ui-v2.py
```

This will launch the web interface at http://localhost:8000 by default.

## How It Works

### Server Connections

The system automatically connects to three MCP servers on startup:

1. **Document Server**: Handles document storage and retrieval
2. **SQLite Server**: Provides database functionality
3. **Markitdown Server**: Converts documents between formats (especially to Markdown)

### Document Workflow

1. Upload a document using the file upload button
2. The system will attempt to convert it to Markdown using the Markitdown server
3. The document is stored in MongoDB with metadata
4. You can later retrieve documents using `/display [document_id]`

### PDF Handling Limitations

The current implementation may have issues converting PDF files to Markdown due to:
- Complex PDF structures (tables, columns, etc.)
- Image-based PDFs or scanned documents
- Required system dependencies in Docker containers


## Extending the System

### Adding New MCP Servers

To add a new MCP server, add its configuration to the `server_configs` list in `chainlit-ui.py`:

```python
{
    "name": "New Server Name",
    "command": "docker",
    "args": [
        "run",
        "--rm",
        "-i",
        # other docker arguments
        "your-mcp-server-image",
        # server arguments
    ],
    "env": None,  # or provide environment variables
}
```

### Switching LLM Providers

The system supports different LLM backends through the model architecture. To use Ollama instead of OpenAI:

```python
model = OllamaModel(model_name="llama2")  # Instead of OpenAI
```

### To view the Uploaded file in Markdown 
When a document is uploaded via the Chainlit UI, a unique UUID is automatically generated for that file.
You can retrieve the UUID by using the Recent Activity tool.

