#!/usr/bin/env python3

import gc

from langchain_ollama import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def do_response(query, messages, model_name, n_ctx):
    messages.append({"role": "user", "content": query})
    model = ChatOllama(
        model=model_name,
        n_ctx=n_ctx,
        temperature=0.0,
        top_p=1.0,
        num_gpu=-1,
        streaming=True,
        keep_alive=0,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    print(f"🤖 {model_name}")
    stream = model.stream(messages)
    response = ""
    for chunk in stream:
        response += chunk.content
    print("\n")
    messages.append({"role": "assistant", "content": response})
    del model
    gc.collect()

instruction = """당신은 친절한 한국어 챗봇입니다.
답변 내용에 한자가 포함되어 있다면 한국어로 번역해서 답변해주세요.
"""
gemma3_history = [{"role": "system", "content": instruction}]

while True:
    question = input("🧑 Question: ")
    question = question.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
    if question.lower() == "bye":
        break

    do_response(
        query=question,
        messages=gemma3_history,
        model_name="gpt-oss:20b",
        n_ctx=131072
    )

