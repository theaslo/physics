"""
Main entry point for the physics MCP server.
"""
from physics.server import mcp

if __name__ == "__main__":
    mcp.run(transport='stdio')
