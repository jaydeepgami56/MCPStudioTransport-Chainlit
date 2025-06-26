import os
import sys
import json
import logging
import uuid
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
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

# Configure more verbose logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG to see all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout instead of stderr
        logging.StreamHandler(sys.stderr),  # Also log to stderr for backup
        # Add a file handler to keep logs in the container
        logging.FileHandler('/app/document_server.log')  
    ]
)
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
    
    def __init__(self, mongo_uri: str, db_name: str, storage_path: str):
        """Initialize the MongoDB document store"""
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.storage_path = storage_path 
        self.client = None
        self.db = None
        self.documents = None
        self.activity_log = None

         # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        self._connect()
        self._init_collections()
        
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
                entities: str, summary: str, original_content: str) -> str:
        """
        Store a document in MongoDB and filesystem
        
        Args:
            content: The document content (markdown version)
            doc_type: Type of document (e.g., "word","txt","pdf","md")
            entities: Key entities extracted from the document as JSON
            summary: Brief summary of the document
            original_content: Original content of the document
            
        Returns:
            Document ID
        """
        logger.info(f"=== DOCUMENT INGESTION STARTED ===")
        logger.info(f"Document type: {doc_type}")
        logger.info(f"Content length: {len(content) if content else 0} chars")
        logger.info(f"Original content length: {len(original_content) if original_content else 0} chars")
        
        # Parse and validate entities
        try:
            # Try to parse as JSON to validate
            if isinstance(entities, str):
                entities_obj = json.loads(entities)
            else:
                # If it's not a string, use as is but convert to string for storage
                entities_obj = entities
                entities = json.dumps(entities)
            
            logger.info("inside document ingestion tool")
            doc_id = str(uuid.uuid4())
            
            # Create document record
            current_time = time.time()
            document = {
                    "uuid": doc_id,
                    "type": doc_type,
                    "entities": entities_obj,
                    "entities_raw": entities,
                    "summary": summary,
                    "content": content,
                    "timestamp": current_time,
                    "formatted_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)),
                    "created_at": current_time,
                    "metadata": {
                        "filename": entities_obj.get("filename", ""),
                        "original_format": entities_obj.get("original_format", "txt"),
                        "original_size": entities_obj.get("original_size", len(content)),
                        "converted_size": len(content),
                        "conversion_status": entities_obj.get("conversion_status", "converted"),
                        "uploaded_at": entities_obj.get("uploaded_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                        "uploaded_by": entities_obj.get("uploaded_by", "user"),
                        "has_markdown": entities_obj.get("has_markdown", True),
                        "markdown_path": entities_obj.get("markdown_path", ""),
                        "original_path": entities_obj.get("storage_path", "")
                    }
                }
            
            # Insert into MongoDB
            result = self.documents.insert_one(document)
            
            # Save MARKDOWN content to proper directory
            markdown_dir = os.path.join(self.storage_path, entities_obj.get("markdown_path", "markdown_files"))
            os.makedirs(markdown_dir, exist_ok=True)
            markdown_file_path = os.path.join(markdown_dir, f"{doc_id}.md")
            with open(markdown_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # ALSO save a copy to the root path for backward compatibility
            root_file_path = os.path.join(self.storage_path, f"{doc_id}.md")
            with open(root_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Save original content if provided
            if original_content:
                # Ensure directories exist
                original_dir = os.path.join(self.storage_path, entities_obj.get("storage_path", "originals"))
                os.makedirs(original_dir, exist_ok=True)
                
                # Get appropriate file extension
                file_ext = entities_obj.get('original_format', 'txt')
                
                # Determine if this is likely a binary file
                binary_extensions = ['pdf', 'docx', 'xlsx', 'pptx', 'zip', 'png', 'jpg', 'jpeg', 'gif']
                is_binary = file_ext.lower() in binary_extensions
                
                # Save the original file
                original_file_path = os.path.join(original_dir, f"{doc_id}.{file_ext}")
                
                if is_binary:
                    # Check if content is base64 encoded (which is likely for binary files sent from client)
                    try:
                        # Try to decode as base64
                        binary_content = base64.b64decode(original_content)
                        with open(original_file_path, 'wb') as f:
                            f.write(binary_content)
                    except:
                        # If not base64 but still binary format, write it as bytes
                        if isinstance(original_content, str):
                            with open(original_file_path, 'wb') as f:
                                f.write(original_content.encode('utf-8'))
                        else:
                            with open(original_file_path, 'wb') as f:
                                f.write(original_content)
                else:
                    # For text files, use text mode
                    with open(original_file_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
            
            # Update document with file paths
            self.documents.update_one(
                {"uuid": doc_id},
                {"$set": {
                    "file_paths": {
                        "markdown": markdown_file_path,
                        "original": original_file_path if original_content else None
                    }
                }}
            )
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a document by its ID
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            Document with metadata
        """
        try:
            doc = self.documents.find_one({"uuid": doc_id})
    
            if not doc:
                print(f"Document not found: {doc_id}")
                return None
            
            # Print document header
            print(f"\n============ DOCUMENT: {doc_id} ============")
            print(f"Type: {doc.get('type', 'Unknown')}")
            print(f"Created: {doc.get('formatted_date', 'Unknown')}")
            print(f"Summary: {doc.get('summary', 'No summary available')}")
            
            # Print metadata
            print("\n--- METADATA ---")
            entities = doc.get('entities', {})
            if entities:
                # Handle entities stored as JSON string
                if isinstance(entities, str):
                    try:
                        entities = json.loads(entities)
                    except:
                        print(f"Could not parse entities JSON: {entities}")
                        entities = {}
                
                # Print each key-value pair
                for key, value in entities.items():
                    print(f"{key}: {value}")
            else:
                print("No metadata available")
            
            # Print full content
            print("\n--- FULL CONTENT ---\n")
            content = doc.get('content', 'No content available')
            print(content)
            
            # Convert ObjectId to string for JSON serialization
            doc["_id"] = str(doc["_id"])
            
            return doc
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            raise

    
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
        self.log_activity("delete", doc_id)
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
        metadata = document.get('metadata', {})
        content = document.get("content", "")
        content_preview = content[:200] + "..." if len(content) > 200 else content
        
        summary = f"📄 Document Summary: {doc_id} 📄\n\n"
        summary += f"Type: {document.get('type', 'Unknown')}\n"
        summary += f"Created: {document.get('formatted_date', 'Unknown')}\n"
        summary += f"Original Format: {metadata.get('original_format', 'Unknown')}\n"
        summary += f"Conversion Status: {metadata.get('conversion_status', 'Unknown')}\n\n"
        summary += f"Summary: {document.get('summary', 'No summary available')}\n\n"
        
        # Add entities if available
        entities = document.get('entities', {})
        if entities:
            summary += "Metadata:\n"
            for key, value in entities.items():
                if key not in ['filename', 'original_format', 'conversion_status']:
                    summary += f"- {key}: {value}\n"
            summary += "\n"
            
        summary += f"Content Preview:\n\n{content_preview}\n"
        
        return summary

    # Add this to the MongoDocumentStore class

    def _init_collections(self):
        """Initialize required collections"""
        # Create activity log collection if it doesn't exist
        if 'activity_log' not in self.db.list_collection_names():
            self.db.create_collection('activity_log')
        self.activity_log = self.db.activity_log

    def log_activity(self, action: str, doc_id: str, details: str = None):
        """
        Log document activity
        
        Args:
            action: Type of action (e.g., "upload", "view", "delete")
            doc_id: ID of the document
            details: Optional details about the action
        """
        current_time = time.time()
        activity = {
            "action": action,
            "doc_id": doc_id,
            "timestamp": current_time,
            "formatted_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        }
        
        if details:
            activity["details"] = details
            
        self.activity_log.insert_one(activity)
        logger.debug(f"Logged {action} activity for document {doc_id}")


async def main(mongo_uri: str, db_name: str, storage_path: str):
    """Main function to start the MCP server"""
    logger.info(f"Starting Document Ingestion MCP Server")
    
    try:
        doc_store = MongoDocumentStore(mongo_uri, db_name, storage_path)
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
                        "    original_content: Original content of the document (optional but recommended).\n\n"
                           "    summary: Summary of the document.\n\n"
                           "Returns:\n"
                           "    The ID of the stored document.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content of the document"},
                        "doc_type": {"type": "string", "description": "Type of document (e.g., 'invoice', 'report')"},
                        "original_content": {"type": "string", "description": "Original content of the document"},
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
            ),
              # Add a new tool for displaying document content
            types.Tool(
                name="display_document",
                description="Display the full content of a document stored in MongoDB.\n\n"
                        "Args:\n"
                        "    doc_id: ID of the document to display.\n\n"
                        "Returns:\n"
                        "    The complete content of the document with metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "ID of the document to display"}
                    },
                    "required": ["doc_id"]
                }
            ),
            
            # Add a tool to get recent activity
            types.Tool(
                name="get_recent_activity",
                description="Retrieve recent document activity (uploads, views, etc.) from MongoDB.\n\n"
                        "Args:\n"
                        "    limit: (Optional) Maximum number of activities to return.\n\n"
                        "Returns:\n"
                        "    A list of recent document activities.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Maximum number of activities to return (default: 10)"}
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
                original_content = arguments.get("original_content", "")  # Get original_content with default empty string
                
                logger.info(f"Ingest document called with: content_length={len(content)}, "
                        f"doc_type={doc_type}, entities_length={len(entities)}, "
                        f"summary_length={len(summary)}, original_content_length={len(original_content)}")
                
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
                    summary=summary,
                    original_content=original_content  # Added the missing parameter
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
            elif name == "display_document":
                if not arguments or "doc_id" not in arguments:
                    raise ValueError("Missing doc_id argument")
                
                doc = doc_store.get_document(arguments["doc_id"])
                
                # Create a formatted display with metadata and content
                display = f"# Document: {doc['uuid']}\n\n"
                display += f"**Type:** {doc['type']}\n"
                display += f"**Created:** {doc.get('formatted_date', 'Unknown')}\n"
                display += f"**Summary:** {doc.get('summary', 'No summary available')}\n\n"
                
                # Add entities if available
                entities = doc.get('entities', {})
                if entities:
                    display += "## Metadata\n\n"
                    for key, value in entities.items():
                        display += f"- **{key}:** {value}\n"
                    display += "\n"
                
                # Add the full content
                display += "## Content\n\n```\n"
                display += doc.get('content', 'No content available')
                display += "\n```\n"
                
                # Log the activity (document view)
                if 'activity_log' in doc_store.db.list_collection_names():
                    doc_store.db.activity_log.insert_one({
                        "action": "view",
                        "doc_id": doc['uuid'],
                        "timestamp": time.time(),
                        "formatted_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    })
                
                return [types.TextContent(type="text", text=display)]

            elif name == "get_recent_activity":
                # Create activity log collection if it doesn't exist
                if 'activity_log' not in doc_store.db.list_collection_names():
                    doc_store.db.create_collection('activity_log')
                
                # Get limit parameter or use default
                limit = 10
                if arguments and "limit" in arguments:
                    limit = int(arguments["limit"])
                
                # Query recent activities
                activities = list(doc_store.db.activity_log.find().sort("timestamp", -1).limit(limit))
                
                # Format activities for display
                activity_display = "# Recent Document Activity\n\n"
                
                if not activities:
                    activity_display += "No recent activity found.\n"
                else:
                    for idx, activity in enumerate(activities, 1):
                        # Convert ObjectId to string for JSON serialization
                        activity["_id"] = str(activity["_id"])
                        
                        # Get document info if available
                        doc_summary = "N/A"
                        try:
                            doc = doc_store.get_document(activity["doc_id"])
                            doc_summary = doc.get('summary', 'No summary')[:50]
                            if len(doc.get('summary', '')) > 50:
                                doc_summary += "..."
                        except:
                            pass
                        
                        # Format the timestamp
                        timestamp = activity.get('formatted_date', 'Unknown')
                        
                        # Add formatted activity entry
                        activity_display += f"### {idx}. {activity.get('action', 'Unknown').capitalize()} - {timestamp}\n"
                        activity_display += f"**Document ID:** `/display {activity.get('doc_id', 'Unknown')}`\n"
                        activity_display += f"**Summary:** {doc_summary}\n\n"
                    
                    activity_display += "---\n\n**Note:** To view a document's full content, type `/display [document-id]` in the chat or click on any Document ID above."
                
                return [types.TextContent(type="text", text=activity_display)]
                    

            
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
    
     # Expand user path
    storage_path = os.path.expanduser(args.storage_path)
    
  
    asyncio.run(main(args.mongo_uri, args.db_name,args.storage_path)) 