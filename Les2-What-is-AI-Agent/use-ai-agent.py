from typing import Union, List
# LangChainのコアに関するライブラリ
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool, tool
from langchain.tools.render import render_text_description

from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from callbacks import AgentCallbackHandler

# LLMに関するライブラリ
from langchain_openai import AzureChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama

# 環境変数に関するライブラリ
from dotenv import load_dotenv
import os
### APIキーの取得
load_dotenv()

@tool
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
    )  # stripping away non alphabetic characters just in case

    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")


if __name__ == "__main__":
    # =====各種設定=====
    # api_key = os.enviton['OPENAI_API_KEY']
    api_key = os.environ['GEMINI_API_KEY']

    #### LLMの初期設定
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",  # or your deployment
        api_version="2024-08-01-preview",
        model='gpt-4o',
        callbacks = [AgentCallbackHandler()],
        stop = ["\nObservation", "Observation"],
    )
    # llm = ChatGoogleGenerativeAI(api_key=api_key, 
    #                              model="gemini-2.0-flash", 
    #                              callbacks = [AgentCallbackHandler()],
                                 
    #                              )
    # llm = ChatOllama(
    #     model="llama3",
    #     stop=["\nObservation", "Observation"],
    #     # callbacks=[AgentCallbackHandler()]
    # )
    
    
    question = "What is the length of the word: DOG"
    tools = [get_text_length]


    # =====STEP1: プロンプトのテンプレート作成と処理=====

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
    
    # LLMに渡すツールの情報を取得
    tool_descriotion = render_text_description(tools)
    print(f"ツールの説明:\n {tool_descriotion}")
    
    tool_names = ", ".join([t.name for t in tools])
    print(f"ツールの名前:\n {tool_names}")

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=tool_descriotion,
        tool_names=tool_names,
    )
    
    FilledPrompt = prompt.format(
        input=question,
        agent_scratchpad=[],
    )

    # print(f"STEP1 LLMに対するプロンプト: {FilledPrompt}")

    
    
    intermediate_steps = []
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # res = agent.invoke({"input":"What is the length of 'DOG' in characters"})
    # print(res)

    # ReAct Agent Loop
    agent_step = None  # Initialize with None instead of empty string
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the word: DOG",
                "agent_scratchpad": intermediate_steps,
            }
        )
        
        # エージェントステップの詳細表示を改善
        print("\n" + "="*50)
        print("🤖 エージェントの動作:")
        print("="*50)
        
        if isinstance(agent_step, AgentAction):
            print(f"📋 アクションタイプ: {type(agent_step).__name__}")
            print(f"🔧 使用ツール: {agent_step.tool}")
            print(f"📥 ツール入力: {agent_step.tool_input}")
            print(f"📝 ログ: \n{agent_step.log}")
            
            # ツールを実行
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            print("\n" + "-"*30)
            print("🛠️ ツール実行中...")
            observation = tool_to_use.func(str(tool_input))
            print(f"📤 結果: {observation}")
            print("-"*30 + "\n")
            
            intermediate_steps.append((agent_step, str(observation)))
        
        elif isinstance(agent_step, AgentFinish):
            print(f"📋 アクションタイプ: {type(agent_step).__name__}")
            print(f"🏁 最終回答: {agent_step.return_values['output']}")
            print(f"📝 ログ: \n{agent_step.log}")
            print("="*50)
            print("### エージェント処理完了 ###")
        else:
            print(f"❓ 未知のアクションタイプ: {type(agent_step)}")