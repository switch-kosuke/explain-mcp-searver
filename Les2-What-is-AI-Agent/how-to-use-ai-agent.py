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


# 環境変数に関するライブラリ
from dotenv import load_dotenv
import os
### APIキーの取得
load_dotenv()

def get_text_length(text: str) -> int:
    """
    テキストの文字数を取得するツール
    Args:
        text (str): 文字数を取得したいテキスト
    Returns:
        int: テキストの文字数
    """
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  
    return len(text)


def get_llm_response(user_quiestion: str) -> str:
    
    llm = ChatOllama(model="llama3")
    # llm = AzureChatOpenAI(
    #     azure_endpoint=os.getenv('api_base'),
    #     openai_api_version=os.getenv('api_version'),
    #     deployment_name=os.getenv('deployment_name'),
    #     openai_api_key=os.getenv('api_key'),
    #     openai_api_type="azure",
    # )

    
    # =====STEP1: エージェントのツールを用意して、REACTエージェントを作成する=====
    # AIエージェント用のツールを用意
    # tools_for_agent = [
    #     Tool(
    #         name="get_text_length", #  ツールの名前(なんでもいい)
    #         func=get_text_length, 
    #         description="useful for when you need get the text length", # ツールの説明(ここを見てLLMはツールを使うか決定する)
    #     )
    # ]

    tools_for_agent = [get_text_length]

    print(tools_for_agent)

    react_prompt = hub.pull("hwchase17/react")
    # create_react_agent: REACTエージェントを作成するための関数
    # llmとツールとプロンプトはこれを使ってね
    agent = create_react_agent(
        llm=llm,
        tools=tools_for_agent,
        prompt=react_prompt
    )

    # =====STEP3: エージェントを実行して、結果を得る=====
    # Agent Executor: エージェントを実行するためのクラス
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=True,
        handle_parsing_errors=True
    )

    result = agent_executor.invoke(
        input={"input": user_quiestion}
    )
    

    print(f"STEP3 LLMの応答: {result['output']}\n")

    
if __name__ == "__main__":
    print(get_llm_response(user_quiestion="What is the length of the word: GOOGLE CLOUD PLATFORM"))
    # print(get_text_length("GOOGLE CLOUD PLATFORM"))