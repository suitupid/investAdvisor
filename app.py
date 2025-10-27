#!/usr/bin/env python3

import atexit
import subprocess

import pandas as pd
import yfinance as yf
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_NAME = 'gpt-oss:20b'
# MODEL_NAME = 'PetrosStav/gemma3-tools:27b'
# MODEL_NAME = 'qwen3:30b'
# MODEL_NAME = 'llama3.3:70b'
# MODEL_NAME = 'llama4:16x17b'

instruction = """
당신은 친절한 한국어 챗봇입니다.
사용자의 질문에 대해 직접적이고 유용한 답변을 한국어로 제공하세요.
간결하면서도 완전한 답변을 작성하세요.
"""

@atexit.register
def stop_model():
    subprocess.run(['ollama', 'stop', MODEL_NAME], check=True)

@tool
def get_stock_price(
    symbol: str,
    start: str,
    end: str,
    interval: str
) -> pd.Series:
    """
    특정 종목(symbol)의 최근 주가, 환율 데이터를 조회하는 도구입니다.
    이 함수는 Yahoo Finance 데이터를 이용해 일정한 간격(interval)으로
    시작 날짜(start)와 마지막 날짜 다음날(end) 사이의 주가 이력을 가져오며,
    그 중 종가(Close) 데이터만 반환합니다.

    ## Parameters
    - **symbol** (`str`): 조회할 종목의 티커(symbol)입니다.  
      예: `"AAPL"`(애플), `"GOOGL"`(구글), `"TSLA"`(테슬라), `"USDKRW=X"`(원달러환율)
    - **start** (`str`): 조회할 시작 날짜입니다.
      날짜 형식은 다음과 같습니다:
        - `"%Y-%m-%d"`
      예: `"2025-10-01"`
    - **end** (`str`): 조회할 마지막 날짜의 다음날입니다.
      날짜 형식은 다음과 같습니다:
        - `"%Y-%m-%d"`
      예: `"2025-10-20"`
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
    # 최근 2025년 10월 1일부터 2025년 10월 20일까지 애플(AAPL) 주가를 1일 간격으로 조회
    get_stock_price(symbol="AAPL", start="2025-10-01", end="2025-10-21", interval="1d")
    ```
    """
    return yf.Ticker(symbol).history(start=start, end=end, interval=interval)['Close']

model = ChatOllama(
    model=MODEL_NAME,
    n_ctx=131072,
    temperature=0.0,
    top_p=1.0,
    num_gpu=-1,
    streaming=True,
    keep_alive=-1,
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

use_tools = ToolNode([get_stock_price])

workflow = StateGraph(MessagesState)
workflow.add_node('model', answer)
workflow.add_node('tools', use_tools)
workflow.set_entry_point('model')
workflow.add_conditional_edges('model', call_tools, ['tools', END])
workflow.add_edge('tools', 'model')
app = workflow.compile()

state = MessagesState({'messages': [SystemMessage(content=instruction)]})
while True:
    while True:
        try:
            user_input = input("🧑 Question: ").strip()
            user_input = user_input.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
            break
        except Exception as e:
            print('🤖 Tell me again.')
    if user_input.lower() == 'bye':
        break
    print(f'🤖 Question: {user_input}')
    state['messages'].append(HumanMessage(content=user_input))
    print('🤖 Response:')
    state = app.invoke(state)
    print()
