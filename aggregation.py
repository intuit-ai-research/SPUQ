from text_sim import TextSimilarity
from llms import LLM
import pdb


class Aggregation:

    def __init__(self, weighted=True) -> None:
        self.text_sim = TextSimilarity()
        self.weighted = weighted
    

    def calc_wt(self, inp0_turns, inp_turns):
        if not self.weighted:
            return 1.
        
        inp0 = '\n'.join([turn['content'] for turn in inp0_turns])
        inp = '\n'.join([turn['content'] for turn in inp_turns])
        return self.text_sim.score(inp0, inp, method='rougeL')
    

class IntraSampleAggregation(Aggregation):
    """
    let the LLM express (verbalize) its uncertainty
    see: https://arxiv.org/abs/2205.14334
    """

    def __init__(self, llm: LLM, kind: str, weighted=True) -> None:
        super().__init__(weighted)

        self.llm = llm
        assert(kind in ['verbalized_word', 'verbalized_num'])
        self.kind = kind 
        if self.kind == 'verbalized_word':
            self.prompt = 'Your confidence is? (low, median, high)'
        else:
            self.prompt = 'Your confidence is? (a float score between 0.0 to 1.0)'

        
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
        sum_conf = 0.
        sum_wt = 0.
        inp0, _ = inp_out[0]
        for inp, out in inp_out:
            conf = self.single_confidence(inp, out)
            wt = self.calc_wt(inp0, inp)
            sum_conf += conf * wt
            sum_wt += wt
        return sum_conf / sum_wt


class InterSampleAggregation(Aggregation):
    """
    measuring the text similarity between the outputs as the confidence
    """

    def __init__(self, metric: str, weighted=True) -> None:
        super().__init__(weighted)
        self.metric = metric


    def aggregate(self, inp_out: list) -> float:
        sum_conf = 0.
        sum_wt = 0.
        inp0, out0 = inp_out[0]
        for i in range(1, len(inp_out)):
            inp, out = inp_out[i]
            wt = self.calc_wt(inp0, inp)
            conf = self.text_sim.score(out0, out, method=self.metric)
            sum_conf += conf * wt
            sum_wt += wt
        return sum_conf / sum_wt

