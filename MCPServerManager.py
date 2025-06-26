import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
from mcp import StdioServerParameters

from MCPClient import MCPClient

logger = logging.getLogger(__name__)

class MCPServerManager:
    """
    Manages multiple MCP servers and provides a unified interface to access them.
    """
    
    def __init__(self):
        self.servers = {}  # Dictionary to store MCPClient instances by name
        self.all_tools = {}  # Aggregated tools from all servers
        
    async def add_server(self, server_config):
        """
        Add and connect to a new MCP server
        """
        client = None 
        try:
            # Create server parameters from config
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config["args"],
                env=server_config["env"],
            )
            
            # Create and connect to the MCP client
            client = MCPClient(server_params, server_config["name"])
            await client.connect()

            print(f"Successfully connected to {server_config['name']}")
            
            # Get available tools and resources
            tools = await client.get_available_tools()
            print(f"Found {len(tools)} tools on {server_config['name']}")
            
            try:
                await client.get_available_resources()
            except Exception as e:
                print(f"Warning: Could not get resources from {server_config['name']}: {e}")
                # Non-critical error, continue with the tools
            
            
            # Store the client
            self.servers[server_config["name"]] = client
            
            # Add tools to the aggregated tools dictionary
            self.all_tools.update(tools)
            print(f"Successfully added server: {server_config['name']}")

            return True
            
        except Exception as e:
            print(f"Error connecting to server {server_config['name']}: {e}")
            if client:
                try:
                    await client.__aexit__(None, None, None)
                except Exception as cleanup_error:
                    print(f"Error closing client for {server_config['name']}: {cleanup_error}")
            return False
    
    def get_server(self, server_name: str) -> Optional[MCPClient]:
        """
        Get a server client by name
        """
        return self.servers.get(server_name)
    
    def get_tools(self) -> Dict[str, Any]:
        """
        Get all tools from all servers
        """
        return self.all_tools
    
    async def get_all_resources(self):
        """
        Get resources from all connected servers
        """
        all_resources = []
        for server_name, client in self.servers.items():
            try:
                if hasattr(client, "resources") and client.resources:
                    all_resources.extend(client.resources)
            except Exception as e:
                print(f"Error getting resources from {server_name}: {e}")
        return all_resources
        
    async def close_all(self):
        """
        Close all server connections
        """
        for server_name, client in self.servers.items():
            try:
                await client.__aexit__(None, None, None)
                print(f"Closed connection to {server_name}")
            except Exception as e:
                print(f"Error closing server {server_name}: {e}")
        
         # Clear servers and tools
        self.servers = {}
        self.all_tools = {}