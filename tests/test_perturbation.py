import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perturbation import TemperaturePerturbation, DummyToken, RandSysMsg, Paraphrasing
import numpy as np
import unittest
import pdb

class TestAggregation(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        
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

        self.messages = [
            {'role': 'user', 'content': 'Is 100 greater than 3?'},
        ]
        self.temperature = 0.7
        self.n = 3
        np.random.seed(2024)


    def test_temperature_perturbation(self):
        perturbation = TemperaturePerturbation(n=self.n)
        perturbed = perturbation.perturb(self.messages, self.temperature)
        print(perturbation)
        print(perturbed)
        self.assertTrue(isinstance(perturbed, list))
        self.assertTrue(len(perturbed) == self.n)
        

    def test_prompt_perturbation(self):
        for _Perturbation in [DummyToken, RandSysMsg, Paraphrasing]:
            perturbation = _Perturbation(n=self.n)
            perturbed = perturbation.perturb(self.messages, self.temperature)
            print(perturbation)
            print(perturbed)
            self.assertTrue(isinstance(perturbed, list))
            self.assertTrue(len(perturbed) == self.n)

if __name__ == '__main__':
    unittest.main()