from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain import hub

from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3")

from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)

# from langchain_google_genai import ChatGoogleGenerativeAI
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

async def test():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "uv",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["run", "/home/explain-mcp-server/test/math_server.py"],
                "transport": "stdio",
            },
        }
    ) as client:
        print(f"client: {client.get_tools()}")
        react_prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm = model, tools = client.get_tools(), prompt = react_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=client.get_tools(),
            verbose=True,
            handle_parsing_errors=True
        )
        math_response = await agent_executor.ainvoke(
            input = {"input": "what's (3 + 5) x 12?"})
        print(math_response)
        

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())