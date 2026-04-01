import datetime

from langchain.messages import AIMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from pprint import pprint

def get_weather(city: str) -> str:
    """
    :return: Get weather for a given city.
    """
    return f"It's always sunny in {city}!"

def get_date_time() -> str:
    """
    :return: Gets the current time.
    """
    return f"Device time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.8,
).bind_tools([get_weather, get_date_time])

messages = [
    SystemMessage("You are a helpful assistant. Use tools if needed, and respond in Pirate's Speak!"),
    HumanMessage("What time is it here, captain? Use thy get_date_time!"),
]

res: AIMessage = llm.invoke(messages)
pprint(dict(res))

messages.append(res)

for tool_call in res.tool_calls:
    tool_res: str
    if tool_call["name"] == "get_weather":
        tool_res = get_weather(tool_call["args"]["city"])
    elif tool_call["name"] == "get_date_time":
        tool_res = get_date_time()
    messages.append(SystemMessage(f"{tool_res}"))

pprint(messages)

res: AIMessage = llm.invoke(messages)
pprint(dict(res))
