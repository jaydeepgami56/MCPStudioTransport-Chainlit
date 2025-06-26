import pymongo
import uuid
import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
from pathlib import Path

# For MCP integration
try:
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    from pydantic import AnyUrl
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not available, some features will be disabled")

# reconfigure UnicodeEncodeError prone default (i.e. windows-1252) to utf-8
if sys.platform == "win32" and os.environ.get('PYTHONIOENCODING') is None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Configure logging
logger = logging.getLogger('mongodb_document_server')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.info("Starting Document Management System with MongoDB")

# MongoDB client and database setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mcp_documents"]
documents_collection = db["documents"]
activity_log_collection = db["activity_log"]

# Ensure activity log collection exists
if 'activity_log' not in db.list_collection_names():
    db.create_collection('activity_log')

# Prompt template for MCP
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
"display_document" - Display the full content of a document.
"get_recent_activity" - Retrieve recent document activity.
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
            self.client = pymongo.MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            # Force a connection check
            self.client.server_info()
            self.db = self.client[self.db_name]
            self.documents = self.db.documents
            logger.info(f"Connected to MongoDB: {self.mongo_uri}, database: {self.db_name}")
        except Exception as e:
            logger.error(f"Could not connect to MongoDB: {e}")
            raise RuntimeError(f"MongoDB connection failed: {e}")
    
    def _init_collections(self):
        """Initialize required collections"""
        # Create activity log collection if it doesn't exist
        if 'activity_log' not in self.db.list_collection_names():
            self.db.create_collection('activity_log')
        self.activity_log = self.db.activity_log
    
    def ingest_document(self, content: str, doc_type: str, 
                        entities: str, summary: str, 
                        original_content: str = None, original_type: str = None) -> str:
        """
        Store a document in MongoDB and filesystem
        
        Args:
            content: The document content (markdown content)
            doc_type: Type of document (e.g., "word","txt","pdf","md")
            entities: Key entities extracted from the document as JSON
            summary: Brief summary of the document
            original_content: The original content of the document (pre-conversion)
            original_type: The original document type
            
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
        
            doc_id = str(uuid.uuid4())
            # Create document record
            current_time = time.time()
            timestamp = datetime.now()
            document = {
                "uuid": doc_id,
                "type": doc_type,
                "entities": entities_obj,
                "entities_raw": entities,
                "summary": summary,
                "content": content,
                "timestamp": current_time,
                "formatted_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)),
                "created_at": timestamp,
                "updated_at": timestamp,
                "has_original": original_content is not None,
                "metadata": {
                    "filename": entities_obj.get("filename", "unknown"),
                    "original_format": entities_obj.get("original_format", "txt"),
                    "original_size": entities_obj.get("original_size", len(content)),
                    "converted_size": entities_obj.get("converted_size", len(content)),
                    "conversion_status": entities_obj.get("conversion_status", "original"),
                    "uploaded_at": entities_obj.get("uploaded_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                    "uploaded_by": entities_obj.get("uploaded_by", "user")
                }
            }
            
            # Add original content if provided
            if original_content:
                document["original_content"] = original_content
                document["original_type"] = original_type or "txt"
            
            # Insert into MongoDB
            result = self.documents.insert_one(document)
            
            # Save content to filesystem
            file_path = os.path.join(self.storage_path, f"{doc_id}.md")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Ingested document {doc_id} of type {doc_type} into MongoDB (ObjectId: {result.inserted_id})")
            self.log_activity("upload", doc_id, f"Uploaded document of type {doc_type}")
            return doc_id
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            raise
    
    def get_document(self, doc_id: str, include_original: bool = False) -> Dict[str, Any]:
        """
        Retrieve a document by its ID
        
        Args:
            doc_id: ID of the document to retrieve
            include_original: Whether to include the original content
            
        Returns:
            Document with metadata
        """
        try:
            document = self.documents.find_one({"uuid": doc_id})
            if not document:
                logger.warning(f"Document {doc_id} not found")
                raise ValueError(f"Document {doc_id} not found")
            
            # Try to get content from filesystem
            file_path = os.path.join(self.storage_path, f"{doc_id}.md")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document['content'] = f.read()
            except FileNotFoundError:
                logger.warning(f"Content file not found for {doc_id}, using stored content")
            
            document["_id"] = str(document["_id"])
            
            # Check if we should return both versions
            if not include_original:
                # Remove original content if present and not requested
                if "original_content" in document and not include_original:
                    del document["original_content"]
            
            self.log_activity("view", doc_id)
            return document
            
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            raise
    
    def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve only the metadata for a document
        
        Args:
            doc_id: The document ID
        
        Returns:
            The document metadata
        """
        document = self.documents.find_one(
            {"uuid": doc_id}, 
            {"_id": 1, "uuid": 1, "metadata": 1, "created_at": 1, "updated_at": 1, 
             "has_original": 1, "type": 1, "summary": 1}
        )
        
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
        
        # Try to delete the file from filesystem
        try:
            file_path = os.path.join(self.storage_path, f"{doc_id}.md")
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete file for {doc_id}: {e}")
        
        # Convert ObjectId to string for JSON serialization
        document["_id"] = str(document["_id"])
        
        logger.info(f"Deleted document {doc_id}")
        self.log_activity("delete", doc_id)
        return {"status": "success", "message": f"Document {doc_id} deleted", "metadata": document}
    
    def list_documents(self, limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
        """
        List documents in the database
        
        Args:
            limit: Maximum number of documents to return
            skip: Number of documents to skip
        
        Returns:
            A list of document summaries
        """
        docs = list(self.documents.find({}, {
            "_id": 1, 
            "uuid": 1, 
            "type": 1, 
            "summary": 1, 
            "timestamp": 1,
            "formatted_date": 1,
            "metadata": 1,
            "created_at": 1,
            "has_original": 1
        }).sort("timestamp", -1).skip(skip).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            if "created_at" in doc and isinstance(doc["created_at"], datetime):
                doc["created_at"] = doc["created_at"].isoformat()
            
        return docs
    
    def search_documents(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents matching criteria
        
        Args:
            query: MongoDB query object
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        # Create text index if it doesn't exist
        try:
            self.documents.create_index([
                ("content", "text"), 
                ("summary", "text")
            ])
        except:
            # Index might already exist
            pass
            
        # If "keyword" is in the query, use text search
        if "keyword" in query and query["keyword"]:
            keyword = query["keyword"]
            text_query = {"$text": {"$search": keyword}}
            projection = {
                "_id": 1, 
                "uuid": 1, 
                "type": 1, 
                "summary": 1, 
                "entities": 1,
                "timestamp": 1,
                "formatted_date": 1,
                "score": {"$meta": "textScore"}
            }
            docs = list(self.documents.find(text_query, projection)
                       .sort([("score", {"$meta": "textScore"})])
                       .limit(limit))
        else:
            # Regular query
            docs = list(self.documents.find(query, {
                "_id": 1, 
                "uuid": 1, 
                "type": 1, 
                "summary": 1, 
                "entities": 1,
                "timestamp": 1,
                "formatted_date": 1
            }).limit(limit))
        
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
        
    def get_recent_activity(self, limit: int = 5) -> str:
        """
        Get recent document activity
        
        Args:
            limit: Maximum number of activities to return
        
        Returns:
            A formatted string of recent activities
        """
        activities = list(self.activity_log.find().sort("timestamp", -1).limit(limit))
        
        if not activities:
            return "# Recent Document Activity\n\nNo recent activity found."
        
        activity_display = "# Recent Document Activity\n\n"
        
        for idx, activity in enumerate(activities, 1):
            # Convert ObjectId to string for JSON serialization
            activity["_id"] = str(activity["_id"])
            
            # Get document info if available
            doc_summary = "N/A"
            try:
                doc = self.get_document(activity["doc_id"])
                doc_summary = doc.get('summary', 'No summary')[:50]
                if len(doc.get('summary', '')) > 50:
                    doc_summary += "..."
            except:
                pass
            
            # Format the timestamp
            timestamp = activity.get('formatted_date', 'Unknown')
            
            # Add formatted activity entry
            activity_display += f"### {idx}. {activity.get('action', 'Unknown').capitalize()} - {timestamp}\n"
            activity_display += f"**Document ID:** {activity.get('doc_id', 'Unknown')}\n"
            activity_display += f"**Summary:** {doc_summary}\n\n"
        
        activity_display += "---\n\n**Note:** To view a document's full content, use the display_document tool with the document ID."
        
        return activity_display


# Legacy function implementations (compatible with original file)
async def ingest_document(content, doc_type="md", entities=None, summary=None, original_content=None, original_type=None):
    """
    Store a document in MongoDB with both original and markdown versions
    
    Args:
        content: The markdown content of the document
        doc_type: The document type, defaults to "md"
        entities: JSON string containing metadata
        summary: A summary of the document
        original_content: The original content of the document (pre-conversion)
        original_type: The original document type
    
    Returns:
        The document ID
    """
    # Use the store initialized in main() if available, otherwise create a temporary one
    global doc_store
    if 'doc_store' not in globals():
        try:
            doc_store = MongoDocumentStore("mongodb://localhost:27017/", "mcp_documents", "~/mcp-documents")
        except Exception as e:
            logger.error(f"Failed to initialize document store: {e}")
            raise
    
    if not summary:
        summary = "No summary provided"
        
    return doc_store.ingest_document(
        content=content,
        doc_type=doc_type,
        entities=entities,
        summary=summary,
        original_content=original_content,
        original_type=original_type
    )

async def display_document(doc_id, include_original=False):
    """
    Retrieve a document from MongoDB by ID
    Optionally include the original content
    
    Args:
        doc_id: The document ID
        include_original: Whether to include the original content
    
    Returns:
        The document content or a JSON object with both versions
    """
    # Use the store initialized in main() if available, otherwise create a temporary one
    global doc_store
    if 'doc_store' not in globals():
        try:
            doc_store = MongoDocumentStore("mongodb://localhost:27017/", "mcp_documents", "~/mcp-documents")
        except Exception as e:
            logger.error(f"Failed to initialize document store: {e}")
            return f"Error retrieving document: {e}"
    
    try:
        document = doc_store.get_document(doc_id, include_original)
        
        # Check if we should return both versions
        if include_original and document.get("has_original", False):
            # Return both versions as JSON
            response_data = {
                "markdown_content": document.get("content", ""),
                "original_content": document.get("original_content", ""),
                "metadata": document.get("metadata", {}),
                "doc_type": document.get("type", "md"),
                "original_type": document.get("original_type", "txt"),
                "summary": document.get("summary", "No summary available")
            }
            return json.dumps(response_data)
        else:
            # Return only the markdown content
            return document.get("content", "")
    except Exception as e:
        return f"Error: {str(e)}"

async def get_document_metadata(doc_id):
    """
    Retrieve only the metadata for a document
    
    Args:
        doc_id: The document ID
    
    Returns:
        The document metadata
    """
    global doc_store
    if 'doc_store' not in globals():
        try:
            doc_store = MongoDocumentStore("mongodb://localhost:27017/", "mcp_documents", "~/mcp-documents")
        except Exception as e:
            logger.error(f"Failed to initialize document store: {e}")
            return f"Error retrieving document metadata: {e}"
    
    try:
        document = doc_store.get_document_metadata(doc_id)
        return json.dumps(document)
    except Exception as e:
        return f"Document {doc_id} not found"

async def list_documents(limit=10, skip=0):
    """
    List documents in the database
    
    Args:
        limit: Maximum number of documents to return
        skip: Number of documents to skip
    
    Returns:
        A list of document summaries
    """
    global doc_store
    if 'doc_store' not in globals():
        try:
            doc_store = MongoDocumentStore("mongodb://localhost:27017/", "mcp_documents", "~/mcp-documents")
        except Exception as e:
            logger.error(f"Failed to initialize document store: {e}")
            return f"Error listing documents: {e}"
    
    try:
        documents = doc_store.list_documents(limit, skip)
        return json.dumps(documents)
    except Exception as e:
        return f"Error: {str(e)}"

async def search_documents(query, limit=10):
    """
    Search documents by content or metadata
    
    Args:
        query: The search query or query object
        limit: Maximum number of documents to return
    
    Returns:
        A list of matching documents
    """
    global doc_store
    if 'doc_store' not in globals():
        try:
            doc_store = MongoDocumentStore("mongodb://localhost:27017/", "mcp_documents", "~/mcp-documents")
        except Exception as e:
            logger.error(f"Failed to initialize document store: {e}")
            return f"Error searching documents: {e}"
    
    try:
        # If query is a string, create text search query
        if isinstance(query, str):
            query = {"keyword": query}
        
        documents = doc_store.search_documents(query, limit)
        return json.dumps(documents)
    except Exception as e:
        return f"Error: {str(e)}"

async def delete_document(doc_id):
    """
    Delete a document from MongoDB
    
    Args:
        doc_id: The document ID
    
    Returns:
        Success message
    """
    global doc_store
    if 'doc_store' not in globals():
        try:
            doc_store = MongoDocumentStore("mongodb://localhost:27017/", "mcp_documents", "~/mcp-documents")
        except Exception as e:
            logger.error(f"Failed to initialize document store: {e}")
            return f"Error deleting document: {e}"
    
    try:
        result = doc_store.delete_document(doc_id)
        if "status" in result and result["status"] == "success":
            return f"Document {doc_id} deleted successfully"
        else:
            return f"Document {doc_id} not found"
    except Exception as e:
        return f"Error: {str(e)}"

async def get_recent_activity(limit=5):
    """
    Get recent document activity
    
    Args:
        limit: Maximum number of activities to return
    
    Returns:
        A formatted string of recent activities
    """
    global doc_store
    if 'doc_store' not in globals():
        try:
            doc_store = MongoDocumentStore("mongodb://localhost:27017/", "mcp_documents", "~/mcp-documents")
        except Exception as e:
            logger.error(f"Failed to initialize document store: {e}")
            return f"Error retrieving activity: {e}"
    
    try:
        return doc_store.get_recent_activity(limit)
    except Exception as e:
        return f"Error: {str(e)}"


async def main(mongo_uri: str, db_name: str, storage_path: str):
    """Main function to start the MCP server"""
    logger.info(f"Starting Document Ingestion MCP Server")
    
    if not MCP_AVAILABLE:
        logger.error("MCP is not available. Cannot start server.")
        sys.exit(1)
    
    try:
        global doc_store
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
                        "    summary: Summary of the document.\n"
                        "    original_content: (Optional) Original content before conversion.\n"
                        "    original_type: (Optional) Original document type.\n\n"
                        "Returns:\n"
                        "    The ID of the stored document.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content of the document"},
                        "doc_type": {"type": "string", "description": "Type of document (e.g., 'invoice', 'report')"},
                        "entities": {"type": "string", "description": "Key entities extracted as JSON string"},
                        "summary": {"type": "string", "description": "Brief summary of the document"},
                        "original_content": {"type": "string", "description": "Original content before conversion"},
                        "original_type": {"type": "string", "description": "Original document type"}
                    },
                    "required": ["content", "doc_type", "entities", "summary"]
                }
            ),
            types.Tool(
                name="get_document",
                description="Retrieve a document by its ID from MongoDB.\n\n"
                        "Args:\n"
                        "    doc_id: ID of the document to retrieve.\n"
                        "    include_original: (Optional) Whether to include original content.\n\n"
                        "Returns:\n"
                        "    The retrieved document as a JSON string or an error message if not found.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "ID of the document to retrieve"},
                        "include_original": {"type": "boolean", "description": "Whether to include original content"}
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
                        "Args:\n"
                        "    limit: (Optional) Maximum number of documents to return.\n"
                        "    skip: (Optional) Number of documents to skip.\n\n"
                        "Returns:\n"
                        "    A list of all stored documents as a JSON string.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Maximum number of documents to return"},
                        "skip": {"type": "integer", "description": "Number of documents to skip"}
                    }
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
            types.Tool(
                name="display_document",
                description="Display the full content of a document stored in MongoDB.\n\n"
                        "Args:\n"
                        "    doc_id: ID of the document to display.\n"
                        "    include_original: (Optional) Whether to include original content.\n\n"
                        "Returns:\n"
                        "    The complete content of the document with metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "ID of the document to display"},
                        "include_original": {"type": "boolean", "description": "Whether to include original content"}
                    },
                    "required": ["doc_id"]
                }
            ),
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
                        "limit": {"type": "integer", "description": "Maximum number of activities to return (default: 5)"}
                    }
                }
            ),
            types.Tool(
                name="get_document_metadata",
                description="Retrieve only the metadata for a document.\n\n"
                        "Args:\n"
                        "    doc_id: ID of the document.\n\n"
                        "Returns:\n"
                        "    The document metadata as a JSON string.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "ID of the document"}
                    },
                    "required": ["doc_id"]
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
                original_content = arguments.get("original_content")
                original_type = arguments.get("original_type")
                
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
                    original_content=original_content,
                    original_type=original_type
                )
                
                # Notify clients that the catalog resource has changed
                await server.request_context.session.send_resource_updated(AnyUrl("doc://catalog"))
                
                return [types.TextContent(type="text", text=f"Document successfully ingested with ID: {doc_id}")]
            
            elif name == "get_document":
                if not arguments or "doc_id" not in arguments:
                    raise ValueError("Missing doc_id argument")
                
                include_original = arguments.get("include_original", False)
                doc = doc_store.get_document(arguments["doc_id"], include_original)
                return [types.TextContent(type="text", text=json.dumps(doc, indent=2))]
            
            elif name == "get_document_metadata":
                if not arguments or "doc_id" not in arguments:
                    raise ValueError("Missing doc_id argument")
                
                metadata = doc_store.get_document_metadata(arguments["doc_id"])
                return [types.TextContent(type="text", text=json.dumps(metadata, indent=2))]
            
            elif name == "delete_document":
                if not arguments or "doc_id" not in arguments:
                    raise ValueError("Missing doc_id argument")
                
                result = doc_store.delete_document(arguments["doc_id"])
                
                # Notify clients that the catalog resource has changed
                await server.request_context.session.send_resource_updated(AnyUrl("doc://catalog"))
                
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
            elif name == "list_documents":
                limit = arguments.get("limit", 10) if arguments else 10
                skip = arguments.get("skip", 0) if arguments else 0
                
                try:
                    limit = int(limit)
                    skip = int(skip)
                except (ValueError, TypeError):
                    limit = 10
                    skip = 0
                    
                docs = doc_store.list_documents(limit, skip)
                return [types.TextContent(type="text", text=json.dumps(docs, indent=2))]
            
            elif name == "search_documents":
                if not arguments:
                    arguments = {}
                
                # Build MongoDB query
                query = {}
                limit = arguments.get("limit", 10)
                
                if "type" in arguments and arguments["type"]:
                    query["type"] = arguments["type"]
                
                if "keyword" in arguments and arguments["keyword"]:
                    keyword = arguments["keyword"]
                    # For text search, we'll use the implementation from the document store
                    query["keyword"] = keyword
                
                if "entity_key" in arguments and arguments["entity_key"] and "entity_value" in arguments:
                    entity_key = arguments["entity_key"]
                    entity_value = arguments["entity_value"]
                    query[f"entities.{entity_key}"] = entity_value
                
                docs = doc_store.search_documents(query, limit)
                return [types.TextContent(type="text", text=json.dumps(docs, indent=2))]
                
            elif name == "display_document":
                if not arguments or "doc_id" not in arguments:
                    raise ValueError("Missing doc_id argument")
                
                include_original = arguments.get("include_original", False)
                doc = doc_store.get_document(arguments["doc_id"], include_original)
                
                # Check if we should return both versions
                if include_original and "original_content" in doc:
                    # Create a formatted display with metadata and both contents
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
                    
                    # Add the converted content
                    display += "## Converted Content\n\n```\n"
                    display += doc.get('content', 'No content available')
                    display += "\n```\n\n"
                    
                    # Add the original content
                    display += "## Original Content\n\n```\n"
                    display += doc.get('original_content', 'No original content available')
                    display += "\n```\n"
                else:
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
                
                return [types.TextContent(type="text", text=display)]

            elif name == "get_recent_activity":
                # Get limit parameter or use default
                limit = arguments.get("limit", 5) if arguments else 5
                
                try:
                    limit = int(limit)
                except (ValueError, TypeError):
                    limit = 5
                    
                activity_text = doc_store.get_recent_activity(limit)
                return [types.TextContent(type="text", text=activity_text)]
            
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
                server_name="document-manager",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

    # Entry point for the script
if __name__ == "__main__":
        import argparse
        
        parser = argparse.ArgumentParser(description="MCP Document Management Server with MongoDB")
        parser.add_argument("--mongo-uri", default="mongodb://localhost:27017", 
                        help="MongoDB connection URI")
        parser.add_argument("--db-name", default="mcp_documents", 
                        help="MongoDB database name")
        parser.add_argument("--storage-path", default="~/mcp-documents", 
                        help="Path to store documents")
        
        args = parser.parse_args()
        
        # Expand user path
        storage_path = os.path.expanduser(args.storage_path)
        
        # Run the main function
        asyncio.run(main(args.mongo_uri, args.db_name, storage_path))