from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain import hub

from langchain_ollama import ChatOllama
# model = ChatOllama(model="llama3")
from langchain_google_genai import ChatGoogleGenerativeAI


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from langchain.agents import AgentExecutor

# from langchain_google_genai import ChatGoogleGenerativeAI
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
# 環境変数に関するライブラリ
from dotenv import load_dotenv
import os
### APIキーの取得
load_dotenv()
async def test():

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    server_params = StdioServerParameters(
        command="/root/.local/bin/uv",
        args=["run", "/home/explain-mcp-server/test/math_server.py"],
    )
        
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            print(agent_response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())