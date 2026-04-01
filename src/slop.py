import datetime
import json
import os
from typing import TypedDict, Optional, Annotated

import dotenv
from langgraph.graph import StateGraph, add_messages
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain.tools import tool

model = "qwen2.5:1.5b"

dotenv.load_dotenv(".env.local")


class Analysis(BaseModel):
    weaknesses: list[str]


class State(TypedDict):
    messages: Annotated[list, add_messages]
    doc_text: str
    analysis: Optional[Analysis]


@tool
def get_weather(city: str) -> str:
    """
    :arg city: name of the city to query
    :return: weather for a given city.
    """
    return f"It's always sunny in {city}!"


@tool
def get_date_time() -> str:
    """
    :return: date and time
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@wrap_tool_call
def tool_exception(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


def load_doc(state: State):
    pdf = PyPDFLoader("res/thebodyshop.pdf")
    doc = pdf.load()
    doc_text = ""
    for (i, page) in enumerate(doc):
        doc_text += f"PAGE {i+1}:\n"
        doc_text += page.page_content + "\n"
    return {"doc_text": doc_text}


def initial_analysis(state: State):
    llm = ChatOllama(
        model=model,
        base_url=str(os.getenv("OLLAMA_BASE_URL")),
        temperature=0.2
    )
    agent = create_agent(
        model=llm,
        name="sailor",
        tools=[get_weather, get_date_time],
        middleware=[tool_exception],
    )
    messages = [
        SystemMessage("Read the legal material provided by the user."
                      "Some information may be irrelevant."
                      "LIST SIX INHERENT WEAKNESSES OR RISKS."
                      "Respond in the same language used in the material."),
        HumanMessage("The legal material:\n\n" + state["doc_text"]),
    ]
    result = agent.invoke({"messages": messages})
    return {"messages": result["messages"]}


def reflection(state: State):
    llm = ChatOllama(
        model=model,
        base_url=str(os.getenv("OLLAMA_BASE_URL")),
        temperature=0.2
    )
    agent = create_agent(
        model=llm,
        name="sailor",
        tools=[get_weather, get_date_time],
        middleware=[tool_exception],
    )
    messages = (state["messages"] + [SystemMessage(
        "Reflect and criticize your analysis: what may be some details that you missed?"
    )])
    result = agent.invoke({"messages": messages})
    return {"messages": result["messages"]}


def improved(state: State):
    llm = ChatOllama(
        model=model,
        base_url=str(os.getenv("OLLAMA_BASE_URL")),
        temperature=0.2
    )
    agent = create_agent(
        model=llm,
        name="sailor",
        tools=[get_weather, get_date_time],
        middleware=[tool_exception],
    )
    messages = state["messages"] + [SystemMessage(
        "Based on your reflection, rewrite an improved version"
        "of your evaluation of the user's material as a JSON object"
        "with a key 'weaknesses' containing an array of strings."
        "Respond in the same language used in the material."
    )]
    result = agent.invoke({"messages": messages})
    return {"messages": result["messages"]}


def output(state: State):
    last_message = state["messages"][-1]
    content = last_message.content.strip()
    if content.startswith("```json"):
        content = content[7:].strip()  # remove ```json
        if content.endswith("```"):
            content = content[:-3].strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Assume it's a list of weaknesses, one per line
        weaknesses = [line.strip() for line in content.split('\n') if line.strip()]
        data = {"weaknesses": weaknesses}
    analysis = Analysis(**data)
    return {"analysis": analysis}



def main():
    builder = StateGraph(State)
    builder.add_node("load_doc", load_doc)
    builder.add_node("initial", initial_analysis)
    builder.add_node("reflect", reflection)
    builder.add_node("improve", improved)
    builder.add_node("output", output)
    builder.add_edge("load_doc", "initial")
    builder.add_edge("initial", "reflect")
    builder.add_edge("reflect", "improve")
    builder.add_edge("improve", "output")
    builder.set_entry_point("load_doc")
    graph = builder.compile()

    result = graph.invoke({})
    print(result["analysis"])


if __name__ == "__main__":
    main()
