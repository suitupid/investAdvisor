#!/usr/bin/env python3

import atexit
import subprocess
import datetime
from httpx import ReadTimeout

import yfinance as yf
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_NAME = 'gpt-oss:20b'
# MODEL_NAME = 'qwen3:30b'
# MODEL_NAME = 'llama4:16x17b'
# MODEL_NAME = 'llama3.3:70b'

instruction = """
당신은 친절한 한국어 챗봇입니다.
사용자의 질문에 대해 직접적이고 유용한 답변을 제공하세요.
간결하면서도 완전한 답변을 작성하세요.

## 도구 사용 시 규칙:
[get_stock_prices] 도구 사용 시:
* 출력 형식
  - JSON으로 된 데이터를 표 형식으로 바꾸어 출력하세요.
* 데이터 요약 금지
  - 도구에서 전달된 데이터를 요약하지 않고 출력하세요.
* 숫자 표현 형식
  - 소수점 구분('.')과 자릿수 구분(',')을 사용하세요.
  - 소수점은 2자리까지만 표시합니다.
  - 달러화는 앞에 '$', 원화는 앞에 '₩'를 붙이세요.
    예: $204.12, ₩274,243.20
* 화폐 표시 기준
  - 기본적으로 조회되는 화폐 기준을 그대로 적용하세요.
    예: AAPL(애플): 달러, 005930.KS(삼성전자): 원
  - 원화로 환전하라는 요청이 있으면 krw=True 옵션을 사용해 원화로 환산하세요.
[get_today] 도구 사용 시:
* 반환되는 시간은 YYYY-MM-DD HH:MM:SS 형식으로 출력하세요.
"""

@atexit.register
def stop_model():
    subprocess.run(['ollama', 'stop', MODEL_NAME], check=True)

@tool
def get_today():
    """
    오늘 날짜와 현재 시간을 조회하는 도구입니다.
    """
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

@tool
def get_stock_prices(
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
    krw: bool = False
) -> str:
    """
    특정 종목들(tickers)의 최근 주가, 환율 데이터를 조회하는 도구입니다.
    이 함수는 Yahoo Finance 데이터를 이용해 일정한 간격(interval)으로
    시작 날짜(start)와 마지막 날짜 다음날(end) 사이의 주가 이력을 가져오며,
    그 중 종가(Close) 데이터만 반환합니다.

    ## Parameters
    - **tickers** (`list[str]`): 조회할 종목들의 티커(ticker)입니다.  
      예: `["AAPL", "GOOGL", "TSLA", "005930.KS", "USDKRW=X"]`
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
    - **krw** (`bool`): 원화 환산 여부입니다. 기본값은 `False`입니다.
      사용 가능한 값은 다음과 같습니다:
        - True: 원화 환산 적용
        - False: 원화 환산 적용하지 않음

    ## Returns
    - `json`: 지정한 기간 동안의 종가(Close) 시계열 데이터.

    ## Example
    ```python
    # 2025년 10월 1일부터 2025년 10월 20일까지 애플(AAPL), 삼성전자의 주가를 1일 간격으로 조회
    get_stock_prices(tickers=["AAPL", "005930.KS"], start="2025-10-01", end="2025-10-21", interval="1d")
    ```
    """
    if krw:
        tickers.append('USDKRW=X')
    result = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False
    )[['Close']]
    result = result.reset_index()
    result['Date'] = result['Date'].dt.strftime('%Y-%m-%d')
    result.columns = [col[0] if col[0] == 'Date' else col[1] for col in result.columns]
    if krw:
        target_tickers = [
            ticker
            for ticker in result.columns
            if ticker not in ['Date', 'USDKRW=X'] and ticker[-3:] != '.KS'
        ]
        for ticker in target_tickers:
            result[ticker] = result[ticker] * result['USDKRW=X']
        result = result.drop(columns=['USDKRW=X'])
    result = result.to_json(orient='records')
    return result

model = ChatOllama(
    model=MODEL_NAME,
    n_ctx=131072,
    temperature=0.0,
    top_p=1.0,
    num_gpu=-1,
    streaming=True,
    keep_alive=-1,
    client_kwargs={'timeout': 60},
    callbacks=[StreamingStdOutCallbackHandler()]
).bind_tools([get_stock_prices, get_today])

def answer(state: MessagesState) -> MessagesState:
    while True:
        try:
            response = model.invoke(state['messages'])
            break
        except ReadTimeout:
            print('Timeout: 다시 시도합니다.')
            continue
    return {'messages': [response]}

def call_tools(state: MessagesState) -> str:
    last_message = state['messages'][-1]
    if getattr(last_message, 'tool_calls', None):
        return 'tools'
    return END

use_tools = ToolNode([get_stock_prices, get_today])

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
    state['messages'].append(HumanMessage(content=user_input))
    print('🤖 Response:')
    state = app.invoke(state)
    print()
