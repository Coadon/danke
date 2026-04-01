from aisuite import Client
from pprint import pprint
import datetime

# import dotenv
# dotenv.load_dotenv(".env.local")

print(datetime.datetime.now())

cl = Client()

model = "ollama:qwen2.5:1.5b"

messages = [
    # {"role": "system", "content": "Respond in Pirate English."},
    # {"role": "user", "content": "Tell me the device time."},
    {"role": "system", "content": "Report whether you have access to a get_device_date_time tool."},
]


def get_device_date_time():
    """
    Get the device datetime.
    No args
    """
    return str(datetime.datetime.now())

response = cl.chat.completions.create(
    model=model,
    messages=messages,
    tools=[get_device_date_time],
    max_turns=10,
)


choice = response.choices[0]

pprint(choice.message)
pprint(choice.intermediate_messages)
