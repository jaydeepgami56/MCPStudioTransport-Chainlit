import os
import sys
import json
import logging
import uuid
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from pydantic import AnyUrl

# MongoDB imports
from pymongo import MongoClient
from bson.objectid import ObjectId
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# reconfigure UnicodeEncodeError prone default (i.e. windows-1252) to utf-8
if sys.platform == "win32" and os.environ.get('PYTHONIOENCODING') is None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger('mcp_document_server')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.info("Starting MCP Document Ingestion Server with MongoDB")

PROMPT_TEMPLATE = """
You are a document management assistant with access to a document repository through MCP.
Your goal is to help the user store, retrieve, and analyze documents using the available tools.

<mcp>
Prompts:
This server provides a "document-assistant" prompt to help users manage their documents. 
It accepts a "mode" argument to determine whether the focus is on ingestion, retrieval, or analysis.

Resources:
"doc://catalog" - A catalog of all documents currently stored in the system.
"doc://summary/{doc_id}" - A summary of the specific document identified by doc_id.

Tools:
"ingest_document" - Store a new document in the repository with metadata.
"get_document" - Retrieve a document by its ID.
"delete_document" - Remove a document from the repository.
"list_documents" - Get a list of all documents in the repository.
"search_documents" - Find documents matching specific criteria.
</mcp>

Based on the chosen mode ({mode}), guide the user through the appropriate document management workflow.

For "ingestion" mode, help them add new documents efficiently with good metadata.
For "retrieval" mode, assist in finding and accessing stored documents.
For "analysis" mode, guide them in extracting insights from their documents.

Remember to be helpful, informative, and focus on the user's document management goals.
"""

class MongoDocumentStore:
    """MongoDB-based document store"""
    
    def __init__(self, mongo_uri: str, db_name: str):
        """Initialize the MongoDB document store"""
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.documents = None
        self._connect()
        
    def _connect(self):
        """Connect to MongoDB"""
        try:
            # Connect to MongoDB with a timeout
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            # Force a connection check
            self.client.server_info()
            self.db = self.client[self.db_name]
            self.documents = self.db.documents
            logger.info(f"Connected to MongoDB: {self.mongo_uri}, database: {self.db_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Could not connect to MongoDB: {e}")
            raise RuntimeError(f"MongoDB connection failed: {e}")
    
    def ingest_document(self, content: str, doc_type: str, 
                        entities: str, summary: str) -> str:
        """
        Store a document in MongoDB
        
        Args:
            content: The document content
            doc_type: Type of document (e.g., "invoice", "report")
            entities: Key entities extracted from the document as JSON
            summary: Brief summary of the document
            
        Returns:
            Document ID
        """
        # Parse and validate entities
        try:
            # Try to parse as JSON to validate
            if isinstance(entities, str):
                entities_obj = json.loads(entities)
            else:
                # If it's not a string, use as is but convert to string for storage
                entities_obj = entities
                entities = json.dumps(entities)
        except json.JSONDecodeError:
            logger.warning(f"Invalid entities JSON. Using empty object.")
            entities = "{}"
            entities_obj = {}
        
        # Create document record
        current_time = time.time()
        document = {
            "uuid": str(uuid.uuid4()),
            "type": doc_type,
            "entities": entities_obj,  # Store as object for better querying
            "entities_raw": entities,  # Store raw JSON string for compatibility
            "summary": summary,
            "content": content,
            "timestamp": current_time,
            "formatted_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)),
            "created_at": current_time
        }
        
        # Insert into MongoDB
        result = self.documents.insert_one(document)
        doc_id = document["uuid"]
        
        logger.info(f"Ingested document {doc_id} of type {doc_type} into MongoDB (ObjectId: {result.inserted_id})")
        return doc_id
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a document by its ID
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            Document with metadata
        """
        document = self.documents.find_one({"uuid": doc_id})
        
        if not document:
            logger.warning(f"Document {doc_id} not found")
            raise ValueError(f"Document {doc_id} not found")
        
        # Convert ObjectId to string for JSON serialization
        document["_id"] = str(document["_id"])
        return document
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document by its ID
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            Status message
        """
        document = self.documents.find_one({"uuid": doc_id})
        
        if not document:
            logger.warning(f"Document {doc_id} not found for deletion")
            raise ValueError(f"Document {doc_id} not found")
        
        # Delete document
        self.documents.delete_one({"uuid": doc_id})
        
        # Convert ObjectId to string for JSON serialization
        document["_id"] = str(document["_id"])
        
        logger.info(f"Deleted document {doc_id}")
        return {"status": "success", "message": f"Document {doc_id} deleted", "metadata": document}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the store
        
        Returns:
            List of document metadata
        """
        docs = list(self.documents.find({}, {
            "_id": 1, 
            "uuid": 1, 
            "type": 1, 
            "summary": 1, 
            "timestamp": 1,
            "formatted_date": 1
        }))
        
        # Convert ObjectId to string for JSON serialization
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            
        return docs
    
    def search_documents(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for documents matching criteria
        
        Args:
            query: MongoDB query object
            
        Returns:
            List of matching documents
        """
        docs = list(self.documents.find(query, {
            "_id": 1, 
            "uuid": 1, 
            "type": 1, 
            "summary": 1, 
            "entities": 1,
            "timestamp": 1,
            "formatted_date": 1
        }))
        
        # Convert ObjectId to string for JSON serialization
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            
        return docs
    
    def generate_catalog(self) -> str:
        """
        Generate a catalog of all documents
        
        Returns:
            Formatted catalog text
        """
        doc_count = self.documents.count_documents({})
        
        if doc_count == 0:
            return "Document Catalog: No documents found in the repository."
        
        catalog = "📑 Document Catalog 📑\n\n"
        catalog += f"Total Documents: {doc_count}\n\n"
        
        # Group by document type using MongoDB aggregation
        pipeline = [
            {"$group": {"_id": "$type", "count": {"$sum": 1}, "docs": {"$push": {"id": "$uuid", "summary": "$summary", "timestamp": "$timestamp"}}}}
        ]
        type_groups = list(self.documents.aggregate(pipeline))
        
        # Add document listing by type
        for group in type_groups:
            doc_type = group["_id"] or "Unknown"
            docs = group["docs"]
            
            catalog += f"## {doc_type.capitalize()} ({len(docs)})\n\n"
            
            # Sort by timestamp (newest first)
            sorted_docs = sorted(docs, key=lambda x: x.get("timestamp", 0), reverse=True)
            
            for doc in sorted_docs:
                catalog += f"- [{doc['id']}] {doc.get('summary', 'No summary')}\n"
            catalog += "\n"
        
        return catalog
    
    def generate_summary(self, doc_id: str) -> str:
        """
        Generate a detailed summary for a specific document
        
        Args:
            doc_id: ID of the document
            
        Returns:
            Formatted summary text
        """
        document = self.documents.find_one({"uuid": doc_id})
        
        if not document:
            return f"Error: Document {doc_id} not found."
        
        content = document.get("content", "")
        content_preview = content[:200] + "..." if len(content) > 200 else content
        
        summary = f"📄 Document Summary: {doc_id} 📄\n\n"
        summary += f"Type: {document.get('type', 'Unknown')}\n"
        summary += f"Created: {document.get('formatted_date', 'Unknown')}\n\n"
        summary += f"Summary: {document.get('summary', 'No summary available')}\n\n"
        
        # Add entities if available
        entities = document.get('entities', {})
        if entities:
            summary += "Entities:\n"
            for key, value in entities.items():
                summary += f"- {key}: {value}\n"
            summary += "\n"
            
        summary += f"Content Preview:\n\n{content_preview}\n"
        
        return summary


async def main(mongo_uri: str, db_name: str):
    """Main function to start the MCP server"""
    logger.info(f"Starting Document Ingestion MCP Server with MongoDB: {mongo_uri}, database: {db_name}")
    
    # Initialize MongoDB document store
    try:
        doc_store = MongoDocumentStore(mongo_uri, db_name)
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB document store: {e}")
        sys.exit(1)
        
    server = Server("document-manager")
    
    # Register handlers
    logger.debug("Registering handlers")
    
    @server.list_resources()
    async def handle_list_resources() -> List[types.Resource]:
        """Handler for listing available resources"""
        logger.debug("Handling list_resources request")
        
        resources = [
            types.Resource(
                uri=AnyUrl("doc://catalog"),
                name="Document Catalog",
                description="A catalog of all documents in the repository",
                mimeType="text/plain",
            )
        ]
        
        # Add resources for each document by querying MongoDB
        docs = doc_store.list_documents()
        for doc in docs:
            resources.append(
                types.Resource(
                    uri=AnyUrl(f"doc://summary/{doc['uuid']}"),
                    name=f"Document Summary: {doc['uuid']}",
                    description=f"Summary of document {doc['uuid']}",
                    mimeType="text/plain",
                )
            )
        
        return resources
    
    @server.read_resource()
    async def handle_read_resource(uri: AnyUrl) -> str:
        """Handler for reading resources"""
        logger.debug(f"Handling read_resource request for URI: {uri}")
        
        if uri.scheme != "doc":
            logger.error(f"Unsupported URI scheme: {uri.scheme}")
            raise ValueError(f"Unsupported URI scheme: {uri.scheme}")
        
        path = str(uri).replace("doc://", "")
        
        if path == "catalog":
            return doc_store.generate_catalog()
        
        if path.startswith("summary/"):
            doc_id = path.replace("summary/", "")
            return doc_store.generate_summary(doc_id)
        
        logger.error(f"Unknown resource path: {path}")
        raise ValueError(f"Unknown resource path: {path}")
    
    @server.list_prompts()
    async def handle_list_prompts() -> List[types.Prompt]:
        """Handler for listing available prompts"""
        logger.debug("Handling list_prompts request")
        
        return [
            types.Prompt(
                name="document-assistant",
                description="A prompt to help manage documents in the repository",
                arguments=[
                    types.PromptArgument(
                        name="mode",
                        description="The mode of operation: ingestion, retrieval, or analysis",
                        required=True,
                    )
                ],
            )
        ]
    
    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: Dict[str, str] | None) -> types.GetPromptResult:
        """Handler for retrieving a prompt"""
        logger.debug(f"Handling get_prompt request for {name} with args {arguments}")
        
        if name != "document-assistant":
            logger.error(f"Unknown prompt: {name}")
            raise ValueError(f"Unknown prompt: {name}")
        
        if not arguments or "mode" not in arguments:
            logger.error("Missing required argument: mode")
            raise ValueError("Missing required argument: mode")
        
        mode = arguments["mode"]
        if mode not in ["ingestion", "retrieval", "analysis"]:
            logger.error(f"Invalid mode: {mode}")
            raise ValueError(f"Invalid mode: {mode}. Must be one of: ingestion, retrieval, analysis")
        
        prompt = PROMPT_TEMPLATE.format(mode=mode)
        
        logger.debug(f"Generated prompt template for mode: {mode}")
        return types.GetPromptResult(
            description=f"Document assistant in {mode} mode",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt.strip()),
                )
            ],
        )
    
    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """Handler for listing available tools"""
        return [
            types.Tool(
                name="ingest_document",
                description="Ingest a document, extract information, and store it in MongoDB.\n\n"
                           "Args:\n"
                           "    content: Content of the document.\n"
                           "    doc_type: Type of the document (e.g., word, txt, pdf, xlsx, md).\n"
                           "    entities: Key entities extracted from the document as a JSON string.\n"
                           "    summary: Summary of the document.\n\n"
                           "Returns:\n"
                           "    The ID of the stored document.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content of the document"},
                        "doc_type": {"type": "string", "description": "Type of document (e.g., 'invoice', 'report')"},
                        "entities": {"type": "string", "description": "Key entities extracted as JSON string"},
                        "summary": {"type": "string", "description": "Brief summary of the document"}
                    },
                    "required": ["content", "doc_type", "entities", "summary"]
                }
            ),
            types.Tool(
                name="get_document",
                description="Retrieve a document by its ID from MongoDB.\n\n"
                           "Args:\n"
                           "    doc_id: ID of the document to retrieve.\n\n"
                           "Returns:\n"
                           "    The retrieved document as a JSON string or an error message if not found.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "ID of the document to retrieve"}
                    },
                    "required": ["doc_id"]
                }
            ),
            types.Tool(
                name="delete_document",
                description="Delete a document by its ID from MongoDB.\n\n"
                           "Args:\n"
                           "    doc_id: ID of the document to delete.\n\n"
                           "Returns:\n"
                           "    A success or error message.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "ID of the document to delete"}
                    },
                    "required": ["doc_id"]
                }
            ),
            types.Tool(
                name="list_documents",
                description="List all documents stored in MongoDB.\n\n"
                           "Returns:\n"
                           "    A list of all stored document IDs as a JSON string.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="search_documents",
                description="Search for documents in MongoDB based on criteria.\n\n"
                           "Args:\n"
                           "    type: (Optional) Type of documents to find.\n"
                           "    keyword: (Optional) Keyword to search in content and summary.\n"
                           "    entity_key: (Optional) Entity field name to search.\n"
                           "    entity_value: (Optional) Entity value to match.\n\n"
                           "Returns:\n"
                           "    A list of matching documents as a JSON string.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "description": "Type of documents to find"},
                        "keyword": {"type": "string", "description": "Keyword to search in content and summary"},
                        "entity_key": {"type": "string", "description": "Entity field name to search"},
                        "entity_value": {"type": "string", "description": "Entity value to match"}
                    }
                }
            )
        ]
    
    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any] | None
    ) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handler for executing tools"""
        try:
            if name == "ingest_document":
                if not arguments or not all(k in arguments for k in ["content", "doc_type", "entities", "summary"]):
                    raise ValueError("Missing required arguments for ingest_document")
                
                # Ensure all arguments are properly validated
                content = arguments["content"]
                doc_type = arguments["doc_type"]
                entities = arguments["entities"]
                summary = arguments["summary"]
                
                # Validate content
                if not isinstance(content, str):
                    logger.warning("Content is not a string, converting...")
                    content = str(content)
                
                # Normalize doc_type
                if not doc_type:
                    doc_type = "unknown"
                
                # Process and ingest the document
                doc_id = doc_store.ingest_document(
                    content=content,
                    doc_type=doc_type,
                    entities=entities,
                    summary=summary
                )
                
                # Notify clients that the catalog resource has changed
                await server.request_context.session.send_resource_updated(AnyUrl("doc://catalog"))
                
                return [types.TextContent(type="text", text=doc_id)]
            
            elif name == "get_document":
                if not arguments or "doc_id" not in arguments:
                    raise ValueError("Missing doc_id argument")
                
                doc = doc_store.get_document(arguments["doc_id"])
                return [types.TextContent(type="text", text=json.dumps(doc, indent=2))]
            
            elif name == "delete_document":
                if not arguments or "doc_id" not in arguments:
                    raise ValueError("Missing doc_id argument")
                
                result = doc_store.delete_document(arguments["doc_id"])
                
                # Notify clients that the catalog resource has changed
                await server.request_context.session.send_resource_updated(AnyUrl("doc://catalog"))
                
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "list_documents":
                docs = doc_store.list_documents()
                return [types.TextContent(type="text", text=json.dumps(docs, indent=2))]
            
            elif name == "search_documents":
                if not arguments:
                    arguments = {}
                
                # Build MongoDB query
                query = {}
                
                if "type" in arguments and arguments["type"]:
                    query["type"] = arguments["type"]
                
                if "keyword" in arguments and arguments["keyword"]:
                    keyword = arguments["keyword"]
                    # Text search in content and summary
                    query["$or"] = [
                        {"content": {"$regex": keyword, "$options": "i"}},
                        {"summary": {"$regex": keyword, "$options": "i"}}
                    ]
                
                if "entity_key" in arguments and arguments["entity_key"] and "entity_value" in arguments:
                    entity_key = arguments["entity_key"]
                    entity_value = arguments["entity_value"]
                    query[f"entities.{entity_key}"] = entity_value
                
                docs = doc_store.search_documents(query)
                return [types.TextContent(type="text", text=json.dumps(docs, indent=2))]
            
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        except Exception as e:
            logger.error(f"Error handling tool call {name}: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    # Start the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("Server running with stdio transport")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="document-ingestion",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

# Entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Document Ingestion Server with MongoDB")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017", 
                      help="MongoDB connection URI")
    parser.add_argument("--db-name", default="mcp_documents", 
                      help="MongoDB database name")
    parser.add_argument("--storage-path", default="~/mcp-documents", help="Path to store documents")
    
    args = parser.parse_args()
    
    import asyncio
    asyncio.run(main(args.mongo_uri, args.db_name)) 