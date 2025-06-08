from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict, Optional
from contextlib import AsyncExitStack
import json
import asyncio
import os

load_dotenv()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class MCP_ChatBot:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.client = genai.Client(api_key=os.getenv("API_KEY"))
        
        # Tools list required for Gemini API
        self.available_tools: List[ToolDefinition] = []
        # Prompts list for quick display 
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions = {}

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server and catalog all its capabilities."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            
            try:
                # List available tools
                response = await session.list_tools()
                for tool in response.tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
                
                # List available prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments if hasattr(prompt, 'arguments') else []
                        })
                
                # List available resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
                
                print(f"✅ Connected to {server_name}")
                
            except Exception as e:
                print(f"Error listing capabilities for {server_name}: {e}")
                
        except Exception as e:
            print(f"❌ Error connecting to {server_name}: {e}")

    async def connect_to_servers(self) -> None:
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
                
            print(f"\n🌐 Total tools available: {len(self.available_tools)}")
            print(f"💡 Total prompts available: {len(self.available_prompts)}")
            
        except Exception as e:
            print(f"❌ Error loading server config: {e}")
            raise

    def _clean_schema(self, schema) -> dict:
        """Recursively clean schema to remove unsupported fields."""
        if not isinstance(schema, dict):
            return schema
        
        forbidden_fields = {
            "title", "additionalProperties", "$schema", "exclusiveMaximum",
            "exclusiveMinimum", "default", "examples", "const", "enum",
            "format", "pattern", "multipleOf", "maximum", "minimum"
        }
        
        cleaned = {}
        for key, value in schema.items():
            if key in forbidden_fields:
                continue
            elif key == "properties" and isinstance(value, dict):
                cleaned[key] = {
                    prop_name: self._clean_schema(prop_schema)
                    for prop_name, prop_schema in value.items()
                }
            elif isinstance(value, dict):
                cleaned[key] = self._clean_schema(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    self._clean_schema(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                cleaned[key] = value
        
        return cleaned

    def _prepare_tools_for_gemini(self) -> List[types.Tool]:
        """Convert MCP tools to Gemini format."""
        return [
            types.Tool(
                function_declarations=[
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": self._clean_schema(tool["input_schema"])
                    }
                ]
            )
            for tool in self.available_tools
        ]

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool and return formatted result."""
        try:
            print(f"🔧 Executing {tool_name} with args: {tool_args}")
            
            session = self.sessions.get(tool_name)
            if not session:
                return f"Tool '{tool_name}' not found."
            
            result = await session.call_tool(tool_name, arguments=tool_args)
            
            # Process result based on type
            if hasattr(result, 'content'):
                if result.content is None:
                    return "No content returned"
                elif isinstance(result.content, list):
                    if len(result.content) > 0:
                        first_item = result.content[0]
                        if hasattr(first_item, 'text'):
                            return first_item.text
                        else:
                            return str(first_item)
                    else:
                        return "Empty result"
                else:
                    return str(result.content)
            else:
                return str(result)
                
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {e}"
            print(f"❌ {error_msg}")
            return error_msg

    async def process_query(self, query: str) -> None:
        """Process a query using Gemini with tool support."""
        tools = self._prepare_tools_for_gemini()
        
        response = await self.client.aio.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=query,
            config=types.GenerateContentConfig(
                temperature=0.1,
                tools=tools,
            ),
        )

        conversation_history = [query]
        process_query = True
        iteration_count = 0
        max_iterations = 10
        
        while process_query and iteration_count < max_iterations:
            iteration_count += 1
            
            if not response.candidates or len(response.candidates) == 0:
                break
                
            candidate = response.candidates[0]
            if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
                break
                
            has_function_call = False
            
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    print(f"\n💬 {part.text}")
                    if len(candidate.content.parts) == 1:
                        process_query = False
                
                elif hasattr(part, 'function_call'):
                    has_function_call = True
                    function_call = part.function_call
                    tool_name = function_call.name
                    tool_args = dict(function_call.args)
                    
                    result_str = await self._execute_tool(tool_name, tool_args)
                    
                    conversation_history.extend([
                        types.Content(parts=[part]),
                        types.Content(
                            role="function",
                            parts=[types.Part(
                                function_response=types.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result_str}
                                )
                            )]
                        )
                    ])
                    
                    response = await self.client.aio.models.generate_content(
                        model="gemini-2.0-flash-lite",
                        contents=conversation_history,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            tools=tools,
                        ),
                    )
            
            if not has_function_call:
                process_query = False

    async def get_resource(self, resource_uri: str) -> None:
        """Fetch and display a resource."""
        session = self.sessions.get(resource_uri)
        
        # Fallback for papers URIs - try any papers resource session
        if not session and resource_uri.startswith("papers://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break
        
        if not session:
            print(f"❌ Resource '{resource_uri}' not found.")
            return
        
        try:
            print(f"📁 Fetching resource: {resource_uri}")
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\n📄 Resource: {resource_uri}")
                print("=" * 50)
                print(result.contents[0].text)
                print("=" * 50)
            else:
                print("❌ No content available.")
        except Exception as e:
            print(f"❌ Error fetching resource: {e}")

    async def list_prompts(self) -> None:
        """List all available prompts."""
        if not self.available_prompts:
            print("❌ No prompts available.")
            return
        
        print("\n💡 Available prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    print(f"    - {arg_name}")

    async def execute_prompt(self, prompt_name: str, args: dict) -> None:
        """Execute a prompt with the given arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"❌ Prompt '{prompt_name}' not found.")
            return
        
        try:
            print(f"💡 Executing prompt '{prompt_name}' with args: {args}")
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content
                
                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(item.text if hasattr(item, 'text') else str(item) 
                                  for item in prompt_content)
                
                print(f"\n🎯 Generated prompt content:")
                print("=" * 50)
                print(text)
                print("=" * 50)
                print("\n🚀 Executing this prompt...")
                await self.process_query(text)
            else:
                print("❌ No prompt content generated.")
        except Exception as e:
            print(f"❌ Error executing prompt: {e}")

    async def chat_loop(self) -> None:
        """Interactive chat loop with enhanced capabilities."""
        print("\n🤖 Enhanced MCP Chatbot Started!")
        print("\n📋 Available Commands:")
        print("  Regular chat: Just type your question")
        print("  @folders     : List available research topics")
        print("  @<topic>     : View papers for a specific topic")
        print("  /prompts     : List available prompts")
        print("  /prompt <name> <arg1=value1> : Execute a prompt")
        print("  quit         : Exit the chatbot")
        print("\n💡 Examples:")
        print("  - search for papers on quantum computing")
        print("  - @folders")
        print("  - @machine_learning")
        print("  - /prompts")
        print("  - /prompt generate_search_prompt topic=AI num_papers=10")
        
        while True:
            try:
                query = input("\n🎯 Query: ").strip()
                if not query:
                    continue
        
                if query.lower() == 'quit':
                    break
                
                # Check for @resource syntax first
                if query.startswith('@'):
                    # Remove @ sign  
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
                    continue
                
                # Check for /command syntax
                if query.startswith('/'):
                    parts = query.split()
                    command = parts[0].lower()
                    
                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("❌ Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue
                        
                        prompt_name = parts[1]
                        args = {}
                        
                        # Parse arguments
                        for arg in parts[2:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key] = value
                        
                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"❌ Unknown command: {command}")
                    continue
                
                # Regular query processing
                await self.process_query(query)
                    
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()

    async def cleanup(self) -> None:
        """Cleanly close all resources."""
        await self.exit_stack.aclose()

async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())