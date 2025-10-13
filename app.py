#!/usr/bin/env python3

import yfinance as yf
from langgraph.graph import StateGraph, ChatState, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

@tool
def get_stock_price(symbol: str) -> str:
    """주식 가격을 조회합니다."""
    ticker = yf.Ticker(symbol)
    result = ticker.history(period="1d")
    if result.empty:
        return f"{symbol} 주식 정보를 찾을 수 없습니다."
    return f"{symbol}: ${round(result['Close'].iloc[0], 2)}"

def answer(state: ChatState) -> ChatState:
    response = model.invoke(state['messages'])
    return {'messages': state['messages'] + [response]}


def should_continue(state: ChatState) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_messages.tool_calls:
        return 'tools'
    return END

model = ChatOllama(
    model='gpt-oss:20b',
    n_ctx=131072,
    temperature=0.0,
    top_p=1.0,
    num_gpu=-1,
    streaming=True,
    keep_alive=0,
    callbacks=[StreamingStdOutCallbackHandler()]
).bind_tools([get_stock_price])

tool_node = ToolNode([get_stock_price])
workflow = StateGraph(MessagesState)

workflow.add_node("model", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("model")

workflow.add_conditional_edges(
    "direct_answer",
    should_continue,
    ["tools", "force_final_answer", END]
)


workflow.add_edge("model", "tools")
workflow.add_edge("tools", "model")

app = workflow.compile()

while True:
    user_input = input("🧑 사용자: ").strip()
    user_input = user_input.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    if user_input.lower() == 'bye':
        break

    # 초기 메시지 생성
    result = app.invoke({"messages": [HumanMessage(content=user_input)]})

    # 결과 출력
    print("\n🤖 모델 응답:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)
        elif isinstance(msg, ToolMessage):
            print(f"[Tool 실행 결과] {msg.content}")
