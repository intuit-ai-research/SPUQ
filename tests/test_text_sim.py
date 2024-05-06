import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_sim import TextSimilarity
import unittest

class TestTextSimilarity(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
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

        self.text_sim = TextSimilarity()
        self.a = 'this is one sentence.'
        self.b = 'it is a sentence.'
        self.c = 'how are you?'

    def test(self):
        for method in ['rouge1', 'rouge2', 'rougeL', 'sbert', 'bertscore']:
            score_ab = self.text_sim.score(self.a, self.b, method=method)
            score_ac = self.text_sim.score(self.a, self.c, method=method)
            print('%s: score_ab = %.4f, score_ac = %.4f'%(method, score_ab, score_ac))
            self.assertTrue(score_ab >= score_ac)


if __name__ == '__main__':
    unittest.main()