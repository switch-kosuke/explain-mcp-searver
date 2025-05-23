# LangChainのコアに関するライブラリ
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLMに関するライブラリ
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# 環境変数に関するライブラリ
from dotenv import load_dotenv
import os
### APIキーの取得
load_dotenv()

information = """
    ウォルト・ディズニー（Walt Disney、1901年12月5日 -1966年12月15日 ）は、アメリカ合衆国・イリノイ州シカゴに生まれたアニメーション作家、アニメーター、プロデューサー、映画監督、脚本家、漫画家、声優、実業家、エンターテイナー。

    ウォルト・ディズニーのサイン
    世界的に有名なアニメーションキャラクター「ミッキーマウス」をはじめとするキャラクターの生みの親で、『ディズニーリゾート』の創立者である。兄のロイ・O・ディズニーと共同で設立したウォルト・ディズニー・カンパニーは数々の倒産、失敗を繰り返すも、350億ドル以上の収入を持つ国際的な大企業に発展した。

    本名はウォルター・イライアス・ディズニー（Walter Elias Disney）。一族はアイルランドからの移民であり、姓の「ディズニー」（Disney）は元々「d'Isigny」と綴られ、フランスのノルマンディー地方のカルヴァドス県のイジニー＝シュル＝メール（フランス語版）から11世紀にイギリスやアイルランドに渡来したノルマン人の末裔であることに由来し、後に英語風に直され「ディズニー」となった。「イライアス」は父名。
"""

def get_llm_response():
        # =====各種設定=====
    # api_key = os.enviton['OPENAI_API_KEY']
    # api_key = os.environ['GEMINI_API_KEY']

    #### LLMの初期設定
    # llm = ChatOpenAI(temperature=0, model_name="gpt-.5-turbo")
    llm = ChatOllama(model="llama3")

    # =====STEP: chainを用いた実行=====
    summary_template = """
    情報を基に、人物像を日本語で要約してください。
    情報：{information}
    """

    summary_prompt_template = PromptTemplate(
        template=summary_template, input_variables="information"
    )

    chain = (
        summary_prompt_template
        | llm 
        | StrOutputParser()
    )

    res = chain.invoke(input={"information": information})

    #### 応答
    print(res)


if __name__=="__main__":
    get_llm_response()