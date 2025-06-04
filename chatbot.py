from dotenv import load_dotenv
from typing import List
import asyncio
from google import genai
from fastmcp import Client
import os

load_dotenv()

class MCP_ChatBot:

    def __init__(self):
        API_KEY = os.getenv("API_KEY")
        # Initialize session and client objects
        self.client = Client("server.py")
        self.gemini = genai.Client(api_key=API_KEY)

    async def process_query(self, query):
        async with self.client:
            response = await self.gemini.aio.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=query,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[self.client.session],  # Pass the FastMCP client session directly
                ),
            )
            
            process_query = True
            while process_query:
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                print(part.text)
                                if len(candidate.content.parts) == 1:
                                    process_query = False
                            
                            elif hasattr(part, 'function_call'):
                                function_call = part.function_call
                                tool_name = function_call.name
                                tool_args = dict(function_call.args)
                                
                                print(f"Calling tool {tool_name} with args {tool_args}")
                                
                                # Call tool through FastMCP
                                result = await self.client.call_tool(tool_name, tool_args)
                                
                                # Continue the conversation with the result
                                response = await self.gemini.aio.models.generate_content(
                                    model="gemini-2.0-flash-lite",
                                    contents=[
                                        query,  # Original query
                                        genai.types.Content(
                                            parts=[part]  # Tool use part
                                        ),
                                        genai.types.Content(
                                            role="function",
                                            parts=[genai.types.Part(
                                                function_response=genai.types.FunctionResponse(
                                                    name=tool_name,
                                                    response={"result": str(result)}
                                                )
                                            )]
                                        )
                                    ],
                                    config=genai.types.GenerateContentConfig(
                                        temperature=0,
                                        tools=[self.client.session],
                                    ),
                                )
                    else:
                        print("No content parts found")
                        process_query = False
                else:
                    print("No candidates in response")
                    process_query = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")

async def main():
    chatbot = MCP_ChatBot()
    await chatbot.chat_loop()

if __name__ == "__main__":
    asyncio.run(main())