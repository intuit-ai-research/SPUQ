import numpy as np
import json, pdb
from llms import LLM
from copy import deepcopy


class Perturbation:
    """
    the (messages, temperature) is perturbed to get n variants
    """
    
    def __init__(self, n) -> None:
        self.n = n

    def perturb(self, messages, temperature) -> list:
        return [(messages, temperature)] * self.n


class TemperaturePerturbation(Perturbation):
    """
    The temperature is perturbed with a random change,
    by sampling in the range (T_min, T_max)
    The prompt is not perturbed
    """
    
    def __init__(self, n, T_min=0, T_max=1.,) -> None:
        self.T_min = T_min
        self.T_max = T_max
        self.n = n


    def perturb(self, messages: list, temperature: float) -> list:
        perturbed = []
        for _ in range(self.n):
            temperature = self.T_min + np.random.random() * (self.T_max - self.T_min)
            perturbed.append((messages, temperature))
        return perturbed



class Paraphrasing(Perturbation):
    """
    The prompt is perturbed by being paraphrased using ChatGPT (3.5).
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.llm = LLM()
        cmd = 'Suggest %i ways to paraphrase the text in triple quotes above.'%n
        cmd += '\nIf the original text is a question, please make sure that the your answers are also questions.'
        cmd += '\nProvide your response in JSON format: {"paraphrased":list_of_str}'
        self.cmd = cmd
        

    def paraphrase(self, messages: list) -> list:
        orig = messages[-1]['content']  # only perturb the last message

        out = self.llm.generate([
            {'role': 'user', 'content': '"""\n' + orig + '\n"""\n' + self.cmd}
        ], temperature=0.7)
        if out is None:
            return None
        t0 = out.find('{')
        if t0 < 0:
            return None
        t1 = out.find('}')
        if t1 < t0:
            return None
        try:
            paraphrased = json.loads(out[t0: t1 + 1])['paraphrased']
        except:
            return None
        xx = []

        for _p in paraphrased:
            x = deepcopy(messages)
            x[-1]['content'] = _p
            xx.append(x)
        return xx
        
    def perturb(self, messages: list, temperature: float) -> list:
        paraphrased = self.paraphrase(messages)
        perturbed = []
        for _x in paraphrased:
            perturbed.append((_x, temperature))
        return perturbed



class RandSysMsg(Perturbation):
    """
    The prompt is perturbed by inserting a random system message.
    """
    def __init__(self, n: int) -> None:
        self.sys_msg = [
            'you are a helpful assistant',
            'you are a question-answering assistant',
            'you are a nice assistant',
            'You are a helpful assistant',
            'You are a question-answering assistant',
            'You are a nice assistant',
            'You are a helpful assistant.',
            'You are a question-answering assistant.',
            'You are a nice assistant.',
        ]
        self.n = n
        assert(n <= len(self.sys_msg))


    def perturb(self, messages: list, temperature: float) -> list:
        sys_msgs = np.random.choice(self.sys_msg, self.n, replace=False)
        perturbed = []
        for sys_msg in sys_msgs:
            x = [{'role': 'system', 'content': sys_msg}] + messages
            perturbed.append((x, temperature))
        return perturbed
    

class DummyToken(Perturbation):
    """
    The prompt is perturbed by inserting a random dummy token.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.dummy_tokens = [
            {
                'text': '\n',
                'pos': 'both',
            },
            {
                'text': '\t',
                'pos': 'both',
            },
            {
                'text': ' ',
                'pos': 'both'
            },
            {
                'text': '...',
                'pos': 'both',
            },
            {
                'text': ' um, ',
                'pos': 'before'
            },
            {
                'text': ' uh, ',
                'pos': 'before'
            },
            {
                'text': '?',
                'pos': 'after'
            },
            {
                'text': '??',
                'pos': 'after'
            },
            {
                'text': '\n\n',
                'pos': 'both',
            },
            {
                'text': ' um... ',
                'pos': 'before'
            },
            {
                'text': ' uh... ',
                'pos': 'before'
            },
        ]

    def perturb(self, messages: list, temperature: float) -> list:
        perturbed = []
        dummies = np.random.choice(self.dummy_tokens, self.n, replace=False)
        for dummy in dummies:
            x = deepcopy(messages)
            if dummy['pos'] == 'both':
                if np.random.random() > 0.5:
                    x[-1]['content'] = x[-1]['content'] + dummy['text']
                else:
                    x[-1]['content'] = dummy['text'] + x[-1]['content']
            elif dummy['pos'] == 'before':
                x[-1]['content'] = dummy['text'] + x[-1]['content']
            else:
                x[-1]['content'] = x[-1]['content'] + dummy['text']
            perturbed.append((x, temperature))
        return perturbed
    
