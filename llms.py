import os
from openai import OpenAI


class LLM:
    def __init__(self, model='gpt-3.5-turbo-0301'):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model

    def generate(self, messages, temperature):
        ret = self.client.chat.completions.create(
            messages=messages,
            temperature=temperature,
            model=self.model,
        )
        return ret.choices[0].message.content
