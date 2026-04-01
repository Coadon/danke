import datetime
import os

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain.tools import tool


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


def main():
    llm = ChatOllama(model="qwen2.5:1.5b")
    llm.temperature = 0.2

    the_agent = create_agent(
        model=llm,
        name="sailor",
        tools=[get_weather, get_date_time],
        middleware=[tool_exception]
    )

    messages = [
        SystemMessage("You are a helpful assistant. Use tools if needed, and respond like you're Donald Trump!"),
        HumanMessage("What's the weather at BJ?"),
    ]

    result = the_agent.invoke(
        input = { "messages": messages },
    )

    messages = result["messages"]
    messages.append(SystemMessage("Reflect on your conclusion and improve if necessary."))

    result = the_agent.invoke(
        input = { "messages": messages },
    )

    messages = result["messages"]

    for message in messages:
        message.pretty_print()


if __name__ == "__main__":
    main()
