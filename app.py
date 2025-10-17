#!/usr/bin/env python3

import yfinance as yf
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

instruction = """
사용자의 질문에 대해 직접적이고 유용한 답변을 제공하세요.
간결하면서도 완전한 답변을 작성하세요.
"""

@tool
def get_stock_price(symbol: str) -> str:
    """주식 가격을 조회합니다."""
    ticker = yf.Ticker(symbol)
    result = ticker.history(period="1d")
    if result.empty:
        return f"{symbol} 주식 정보를 찾을 수 없습니다."
    return f"{symbol}: ${round(result['Close'].iloc[0], 2)}"

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

def answer(state: MessagesState) -> MessagesState:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: MessagesState) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'tools'
    return END

tool_node = ToolNode([get_stock_price])

workflow = StateGraph(MessagesState)
workflow.add_node("model", answer)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("model")
workflow.add_conditional_edges("model", call_tools, ["tools", END])
workflow.add_edge("tools", "model")
app = workflow.compile()

state = MessagesState({"messages": [SystemMessage(content=instruction)]})
while True:
    user_input = input("🧑 Question: ").strip()
    user_input = user_input.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    if user_input.lower() == 'bye':
        break
    state["messages"].append(HumanMessage(content=user_input))
    print("🤖 Response:")
    state = app.invoke(state)
    print()

