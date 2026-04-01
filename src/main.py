import datetime
import os
from pprint import pprint
import dotenv


from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_text_splitters import CharacterTextSplitter

model = "qwen2.5:1.5b"

dotenv.load_dotenv(".env.local")

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


def run_doc(doc: list[Document]):
    doc_text = ""
    for (i, page) in enumerate(doc):
        doc_text += f"\npage {i+1}\n"
        doc_text += page.page_content + "\n"

    llm = ChatOllama(
        model=model,
        base_url=str(os.getenv("OLLAMA_BASE_URL")),
        temperature=0.2
    )

    legal_eval_agent = create_agent(
        model=llm,
        name="sailor",
        tools=[get_weather, get_date_time],
        middleware=[tool_exception]
    )

    messages = [
        SystemMessage("Read the legal material provided by the user.\n"
                      "Some information may be irrelevant.\n"
                      "LIST SIX INHERENT WEAKNESSES OR RISKS.\n"
                      "Respond in the same language used in the material."),
        HumanMessage("The legal material:\n\n" + doc_text),
    ]

    result = legal_eval_agent.invoke({ "messages": messages })
    messages = result["messages"]

    messages.append(SystemMessage("Reflect and criticize your analysis: what may be some details that you missed?"))

    result = legal_eval_agent.invoke({ "messages": messages })
    messages = result["messages"]

    messages.append(SystemMessage("Based on your reflection, rewrite an improved version of your evaluation of the user's material."))

    result = legal_eval_agent.invoke({ "messages": messages })
    messages = result["messages"]

    for message in messages:
        message.pretty_print()


def main():
    pdf = PyPDFLoader("res/thebodyshop.pdf")
    doc = pdf.load()
    run_doc(doc)


if __name__ == "__main__":
    main()
