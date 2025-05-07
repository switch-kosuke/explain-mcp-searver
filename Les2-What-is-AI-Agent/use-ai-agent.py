from typing import Union, List
# LangChainのコアに関するライブラリ
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool, tool
from langchain.tools.render import render_text_description

from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
# from callbacks import AgentCallbackHandler

# LLMに関するライブラリ
# from langchain_openai import AzureChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# 環境変数に関するライブラリ
# from dotenv import load_dotenv
# import os
#### APIキーの取得
# load_dotenv()

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
    # api_key = os.environ['GEMINI_API_KEY']

    #### LLMの初期設定
    # llm = ChatOpenAI(temperature=0, model_name="gpt-.5-turbo")
    # llm = ChatGoogleGenerativeAI(api_key=api_key, temperature=0, model="gemini-1.5-flash",max_output_tokens=20)
    llm = ChatOllama(
        model="llama3",
        stop=["\nObservation", "Observation"],
        # callbacks=[AgentCallbackHandler()]
    )


    # =====STEP1: プロンプトのテンプレート作成と処理=====
    tools = [get_text_length]

    template = """
    以下の質問にできる限り答えてください。利用可能なツールは次の通りです：

    {tools}

    次の形式を使用してください：

    質問: あなたが答えるべき入力質問 
    考え: 何をすべきかを常に考える 
    アクション: 取るべきアクション、次の中の1つ [{tool_names}] 
    アクション入力: アクションへの入力 
    観察: アクションの結果 ...（この「考え」「アクション」「アクション入力」「観察」がN回繰り返されることがあります） 
    考え: 最終的な答えが分かった 

    最終的な答え: 元の入力質問に対する最終的な答え

    始めましょう！

    質問: {input}  
    考え: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    FilledPrompt = prompt.format(
        input="What is the length of the word: DOG",
        agent_scratchpad=[],
    )

    print(f"STEP1 LLMに対するプロンプト: {FilledPrompt}")

    # =====STEP2: LLMに投げて結果を得る=====
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

    # # res = agent.invoke({"input":"What is the length of 'DOG' in characters"})
    # # print(res)

    # ## ReAct Agent Loop
    # agent_step = ""
    # while not isinstance(agent_step, AgentFinish):
    #     agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
    #         {
    #             "input": "What is the length of the word: DOG",
    #             "agent_scratchpad": intermediate_steps,
    #         }
    #     )
    #     print(agent_step)

    #     if isinstance(agent_step, AgentAction):
    #         tool_name = agent_step.tool
    #         tool_to_use = find_tool_by_name(tools, tool_name)
    #         tool_input = agent_step.tool_input

    #         observation = tool_to_use.func(str(tool_input))
    #         print(f"{observation=}")
    #         intermediate_steps.append((agent_step, str(observation)))
    
    # if isinstance(agent_step, AgentFinish):
    #     print("### AgentFinish ###")
    #     print(agent_step.return_values)