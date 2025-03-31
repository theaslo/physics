#!/usr/bin/env python
"""
Entry point for the physics MCP server.
This script allows running the server directly.
"""
from physics.server import mcp

def main():
    print("Starting physics MCP server...")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
