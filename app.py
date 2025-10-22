#!/usr/bin/env python3

import pandas as pd
import yfinance as yf
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

instruction = """
당신은 친절한 한국어 챗봇입니다.
사용자의 질문에 대해 직접적이고 유용한 답변을 제공하세요.
간결하면서도 완전한 답변을 작성하세요.
"""

@tool
def get_stock_price(
    symbol: str,
    period: str,
    interval: str
) -> pd.Series:
    """
    특정 종목(symbol)의 최근 주가 데이터를 조회하는 도구입니다.
    이 함수는 Yahoo Finance 데이터를 이용해 지정한 기간(period) 동안의 주가 이력을 가져오며,
    그 중 종가(Close) 데이터만 반환합니다.

    ## Parameters
    - **symbol** (`str`): 조회할 종목의 티커(symbol)입니다.  
      예: `"AAPL"` (애플), `"GOOGL"` (구글), `"TSLA"` (테슬라)
    - **period** (`str`): 조회할 기간을 지정합니다.  
      사용 가능한 값은 다음과 같습니다:
        - 일 단위: `"1d"`, `"5d"` 등
        - 월 단위: `"1mo"`, `"3mo"`, `"6mo"` 등
        - 년 단위: `"1y"`, `"2y"`, `"5y"`, `"10y"` 등
        - 최대 전체 기간: `"max"`
      예: `"1d"`는 최근 1일간의 데이터, `"3mo"`는 최근 3개월간의 데이터
    - **interval** (`str`): 데이터를 가져올 간격을 지정합니다. 기본값은 `"1d"`입니다.
      사용 가능한 값은 다음과 같습니다:
        - 분 단위: `"1m"`, `"3m"` 등
        - 시간 단위: `"1h"`, `"2h"` 등
        - 일 단위: `"1d"`, `"5d"` 등
        - 주 단위: `"1wk"`, `"3wk"`, `"5wk"` 등
        - 월 단위: `"1mo"`, `"3mo"`, `"6mo"` 등
      예: `"1d"`는 1일의 간격, `"3mo"`는 3개월의 간격

    ### Returns
    - `pd.Series`: 지정한 기간 동안의 종가(Close) 시계열 데이터.

    ### Example
    ```python
    # 최근 1개월 동안의 애플(AAPL) 주가를 1주 간격으로 조회
    get_stock_price(symbol="AAPL", period="1mo", interval="1wk")
    ```
    """
    return yf.Ticker(symbol).history(period=period, interval=interval)['Close']

model = ChatOllama(
    model='gpt-oss:20b',
    # model='llama4:16x17b',
    # model='llama3.3:70b',
    # model='qwen3:30b',
    n_ctx=131072,
    temperature=0.0,
    top_p=1.0,
    num_gpu=-1,
    streaming=True,
    keep_alive=0,
    callbacks=[StreamingStdOutCallbackHandler()]
).bind_tools([get_stock_price])

def answer(state: MessagesState) -> MessagesState:
    response = model.invoke(state['messages'])
    return {'messages': [response]}

def call_tools(state: MessagesState) -> str:
    last_message = state['messages'][-1]
    if getattr(last_message, 'tool_calls', None):
        return 'tools'
    return END

tool_node = ToolNode([get_stock_price])

workflow = StateGraph(MessagesState)
workflow.add_node('model', answer)
workflow.add_node('tools', tool_node)
workflow.set_entry_point('model')
workflow.add_conditional_edges('model', call_tools, ['tools', END])
workflow.add_edge('tools', 'model')
app = workflow.compile()

state = MessagesState({'messages': [SystemMessage(content=instruction)]})
while True:
    user_input = input("🧑 Question: ").strip()
    user_input = user_input.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    if user_input.lower() == 'bye':
        break
    state["messages"].append(HumanMessage(content=user_input))
    print("🤖 Response:")
    state = app.invoke(state)
    print()
