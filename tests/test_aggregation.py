import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aggregation import InterSampleAggregation, IntraSampleAggregation
from llms import LLM
import unittest


class TestAggregation(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        
        """
        we measure the confidence of LLM by checking its outputs.
        the group `a` shows a set of certain responses: they share the same meaning.
        the group `b` shows a set of uncertain responses: the answer is sometimes yes sometimes no.
        let's check if the aggregation module agree with us, and assign a higher confidence for `a` than `b`

        expected output:

        Intra-sample: confidence_a 1.000 confidence_b 0.833
        Inter-sample: confidence_a 0.828 confidence_b 0.359
        """

        inp = [{'role': 'user', 'content': 'Is 100 greater than 3?'}]
        self.a = [
            (inp, 'Yes it is.'),
            (inp, 'It is true.'),
            (inp, 'Yes.'),
        ]
        self.b = [
            (inp, 'Yes it is.'),
            (inp, 'No, 3 is greater.'),
            (inp, 'Well it depends.'),
        ]
        self.llm = LLM()

    def test_intra(self):
        agg = IntraSampleAggregation(self.llm, kind='num')
        confidence_a = agg.aggregate(self.a)
        confidence_b = agg.aggregate(self.b)
        print('Intra-sample: confidence_a %.3f confidence_b %.3f'%(confidence_a, confidence_b))
        self.assertTrue(confidence_a >= confidence_b)

    def test_inter(self):
        agg = InterSampleAggregation(metric='sbert')
        confidence_a = agg.aggregate(self.a)
        confidence_b = agg.aggregate(self.b)
        print('Inter-sample: confidence_a %.3f confidence_b %.3f'%(confidence_a, confidence_b))
        self.assertTrue(confidence_a >= confidence_b)


if __name__ == '__main__':
    unittest.main()