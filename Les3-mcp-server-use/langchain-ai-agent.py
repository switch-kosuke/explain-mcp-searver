# LangChainのコアに関するライブラリ
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_core.tools import Tool

# LLMに関するライブラリ
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama

# MCPサーバーに関するライブラリ
import langchain_mcp_adapters.tools
import mcp

# 環境変数に関するライブラリ
from dotenv import load_dotenv
import os
### APIキーの取得
load_dotenv()


llm = ChatOllama(model="llama3")
    # llm = AzureChatOpenAI(
    #     azure_endpoint=os.getenv('api_base'),
    #     openai_api_version=os.getenv('api_version'),
    #     deployment_name=os.getenv('deployment_name'),
    #     openai_api_key=os.getenv('api_key'),
    #     openai_api_type="azure",
    # )

async def get_text_length(text: str):
    # MCP サーバ呼出の設定
    params = mcp.StdioServerParameters(
        command="python",
        args=["server.py"],
    )

    # MCP サーバを実行
    async with mcp.client.stdio.stdio_client(params) as (read, write):
        async with mcp.ClientSession(read, write) as session:
            await session.initialize()
            tools = await langchain_mcp_adapters.tools.load_mcp_tools(session)
        
        
        react_prompt = hub.pull("hwchase17/react")
        # create_react_agent: REACTエージェントを作成するための関数
        # llmとツールとプロンプトはこれを使ってね
        agent = create_react_agent(
            llm=llm,
            tools=client.get_tools(),
            prompt=react_prompt
        )

        # =====STEP3: エージェントを実行して、結果を得る=====
        # Agent Executor: エージェントを実行するためのクラス
        agent_executor = AgentExecutor(
            agent=agent,
            tools=client.get_tools(),
            verbose=True,
            handle_parsing_errors=True
        )

        result = agent_executor.invoke(
            input={"input": "What is the length of the word: GOOGLE CLOUD PLATFORM"}
        )
        

        print(f"STEP3 LLMの応答: {result['output']}\n")

    
if __name__ == "__main__":
    # print(get_llm_response(user_quiestion="What is the length of the word: GOOGLE CLOUD PLATFORM"))
    # print(get_text_length("GOOGLE CLOUD PLATFORM"))
    asyncio.run(get_text_length("GOOGLE CLOUD PLATFORM"))