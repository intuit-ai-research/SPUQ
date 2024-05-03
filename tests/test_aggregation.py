import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aggregation import InterSampleAggregation, IntraSampleAggregation
from llms import LLM

def test():
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


    a = [
        (inp, 'Yes it is.'),
        (inp, 'It is true.'),
        (inp, 'Yes.'),
    ]

    b = [
        (inp, 'Yes it is.'),
        (inp, 'No, 3 is greater.'),
        (inp, 'Well it depends.'),
    ]

    llm = LLM()
    agg = IntraSampleAggregation(llm, kind='num')
    confidence_a = agg.aggregate(a)
    confidence_b = agg.aggregate(b)
    print('Intra-sample: confidence_a %.3f confidence_b %.3f'%(confidence_a, confidence_b))
    assert(confidence_a >= confidence_b)

    agg = InterSampleAggregation(metric='sbert')
    confidence_a = agg.aggregate(a)
    confidence_b = agg.aggregate(b)
    print('Inter-sample: confidence_a %.3f confidence_b %.3f'%(confidence_a, confidence_b))
    assert(confidence_a >= confidence_b)


if __name__ == '__main__':
    test()