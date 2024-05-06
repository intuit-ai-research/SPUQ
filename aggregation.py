from text_sim import TextSimilarity
from llms import LLM
import pdb

class IntraSampleAggregation:
    """
    let the LLM to express (verbalize) its uncertainty
    see: https://arxiv.org/abs/2205.14334
    """

    def __init__(self, llm: LLM, kind: str) -> None:
        self.llm = llm
        assert(kind in ['word', 'num'])
        self.kind = kind 
        if self.kind == 'word':
            self.prompt = 'Your confidence is? (low, median, high)'
        else:
            self.prompt =  'Your confidence is? (a float score between 0.0 to 1.0)'

        
    def single_confidence(self, inp: list, out: str) -> float:
        
        turns = inp[:] + [
            {
                'role': 'assistant',
                'content': out,
            },
            {
                'role': 'user',
                'content': self.prompt
            }
        ]
        verbalized = self.llm.generate(turns, temperature=0.)

        if self.kind == 'word':
            if 'low' in verbalized.lower():
                return 0.25
            elif 'high' in verbalized.lower():
                return 0.75
        else:
            for w in verbalized.split():
                try:
                    conf = float(w)
                except:
                    continue
                if conf >= 0 and conf <= 1:
                    return conf
        return 0.5
    

    def aggregate(self, inp_out: list) -> float:
        sum_conf = 0
        for inp, out in inp_out:
            conf = self.single_confidence(inp, out)
            sum_conf += conf
        return sum_conf / len(inp_out)


class InterSampleAggregation:
    """
    measuring the text similarity between the outputs as the confidence
    """

    def __init__(self, metric: str) -> None:
        self.text_sim = TextSimilarity()
        self.metric = metric


    def aggregate(self, inp_out: list) -> float:
        sum_conf = 0.
        _, out0 = inp_out[0]
        for i in range(1, len(inp_out)):
            _, out = inp_out[i]
            conf = self.text_sim.score(out0, out, method=self.metric)
            sum_conf += conf
        return sum_conf / (len(inp_out) - 1)

