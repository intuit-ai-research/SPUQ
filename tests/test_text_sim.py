import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_sim import TextSimilarity


def test():
    """
    in this example,
    to human, `a` looks more similar to `b` than `c`
    let's make sure the metric agrees with this.
    
    expected output:
    
    rouge1: score_ab = 0.5000, score_ac = 0.0000
    rouge2: score_ab = 0.0000, score_ac = 0.0000
    rougeL: score_ab = 0.5000, score_ac = 0.0000
    sbert: score_ab = 0.7880, score_ac = 0.1007
    bertscore: score_ab = 0.9551, score_ac = 0.8531
    """

    text_sim = TextSimilarity()
    a = 'this is one sentence.'
    b = 'it is a sentence.'
    c = 'how are you?'

    for method in ['rouge1', 'rouge2', 'rougeL', 'sbert', 'bertscore']:
        score_ab = text_sim.score(a, b, method=method)
        score_ac = text_sim.score(a, c, method=method)
        print('%s: score_ab = %.4f, score_ac = %.4f'%(method, score_ab, score_ac))
        assert(score_ab >= score_ac)


if __name__ == '__main__':
    test()