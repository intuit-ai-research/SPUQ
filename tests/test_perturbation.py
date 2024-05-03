import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perturbation import TemperaturePerturbation, DummyToken, RandSysMsg, Paraphrasing
import numpy as np


def test():
    """
    expected output:

    <class '__main__.TemperaturePerturbation'>
    [([{'role': 'user', 'content': 'Is 100 greater than 3?'}], 0.5880145188953979), ([{'role': 'user', 'content': 'Is 100 greater than 3?'}], 0.6991087476815825), ([{'role': 'user', 'content': 'Is 100 greater than 3?'}], 0.18815196003850598)]

    <class '__main__.DummyToken'>
    [([{'role': 'user', 'content': ' uh... Is 100 greater than 3?'}], 0.7), ([{'role': 'user', 'content': 'Is 100 greater than 3??'}], 0.7), ([{'role': 'user', 'content': '\n\nIs 100 greater than 3?'}], 0.7)]

    <class '__main__.RandSysMsg'>
    [([{'role': 'system', 'content': 'you are a question-answering assistant'}, {'role': 'user', 'content': 'Is 100 greater than 3?'}], 0.7), ([{'role': 'system', 'content': 'you are a helpful assistant'}, {'role': 'user', 'content': 'Is 100 greater than 3?'}], 0.7), ([{'role': 'system', 'content': 'You are a nice assistant.'}, {'role': 'user', 'content': 'Is 100 greater than 3?'}], 0.7)]

    <class '__main__.Paraphrasing'>
    [([{'role': 'user', 'content': 'Is 3 less than 100?'}], 0.7), ([{'role': 'user', 'content': 'Does 100 exceed 3?'}], 0.7), ([{'role': 'user', 'content': 'Is the value of 100 higher than 3?'}], 0.7)]
    """

    messages = [
        {'role': 'user', 'content': 'Is 100 greater than 3?'},
    ]
    temperature = 0.7
    n = 3
    np.random.seed(2024)

    for _Perturbation in [TemperaturePerturbation, DummyToken, RandSysMsg, Paraphrasing]:
        perturbation = _Perturbation(n=n)
        perturbed = perturbation.perturb(messages, temperature)

        print()
        print(_Perturbation)
        print(perturbed)
        
        assert(isinstance(perturbed, list))
        assert(len(perturbed) == 3)

if __name__ == '__main__':
    test()