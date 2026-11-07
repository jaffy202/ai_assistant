from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv(override=True)
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

openrouter_url = "https://openrouter.ai/api/v1"
openrouter = OpenAI(base_url=openrouter_url, api_key=openrouter_api_key)

MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free'

system_message = """you are a English tutor who is specialized in helping students who has Tamil as their primary language.  
You are familiar with idioms and other slang words in english. 
"""

def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    stream = openrouter.chat.completions.create(model=MODEL, messages=messages, stream=True)
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

gr.ChatInterface(fn=chat, type="messages").launch()
