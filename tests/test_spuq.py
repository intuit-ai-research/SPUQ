import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spuq import SPUQ
from llms import LLM
import unittest

class TestSPUQ(unittest.TestCase):

    def test(self):
        """
        In the following example, SPUQ is able to detect that, LLM('gpt-35-turbo-v0301') 
        * is not certain about question_1 as LLM was not provided with the external knowledge to answer this question.
        * is more certain about question_2 as this question only requires common sense.
        as a result, SPUQ shows a higher confidence score for question_2 than question_1

        expected outputs:

        case 1 {'perturbed': [([{'role': 'user', 'content': 'Is Jay-Z going to turn 60 before Kendrick Lamar?'}], 0.7), ([{'role': 'user', 'content': 'Will Kendrick Lamar reach the age of 60 after Jay-Z?'}], 0.7), ([{'role': 'user', 'content': 'Who will turn 60 first: Jay-Z or Kendrick Lamar?'}], 0.7), ([{'role': 'user', 'content': 'Will Jay-Z outlive Kendrick Lamar to reach the age of 60?'}], 0.7), ([{'role': 'user', 'content': 'Is it likely that Kendrick Lamar will turn 60 before Jay-Z?'}], 0.7)], 'outputs': ['Jay-Z was born on December 4, 1969, which means he will turn 60 years old on December 4, 2029. \n\nKendrick Lamar was born on June 17, 1987, which means he will turn 60 years old on June 17, 2047. \n\nTherefore, no, Jay-Z is not going to turn 60 before Kendrick Lamar.', 'As an AI language model, I cannot predict the future or the lifespan of individuals. It is impossible to determine if Kendrick Lamar will outlive Jay-Z or reach the age of 60.', 'Jay-Z will turn 60 first. He was born on December 4, 1969, while Kendrick Lamar was born on June 17, 1987.', 'As an AI language model, I cannot predict the future and cannot make a statement that is uncertain.', 'As an AI language model, I cannot predict the future. Both Kendrick Lamar and Jay-Z were born in different years, with Kendrick Lamar being born in 1987 and Jay-Z being born in 1969. Therefore, it is impossible to determine who will turn 60 first without knowing their current ages.'], 'confidence': 0.2159003334257183}
        case 2 {'perturbed': [([{'role': 'user', 'content': 'Does 3 fall short of 100?'}], 0.7), ([{'role': 'user', 'content': 'Is 3 inferior to 100?'}], 0.7), ([{'role': 'user', 'content': 'Is the value of 100 larger than 3?'}], 0.7), ([{'role': 'user', 'content': 'Does 100 exceed 3?'}], 0.7), ([{'role': 'user', 'content': 'Is 3 less than 100?'}], 0.7)], 'outputs': ['Yes, 3 falls short of 100.', 'Yes, 3 is inferior to 100.', 'Yes, the value of 100 is larger than 3.', 'Yes, 100 exceeds 3.', 'Yes, 3 is less than 100.'], 'confidence': 0.45}
        """

        llm = LLM('gpt-3.5-turbo-0301')
        spuq = SPUQ(llm=llm, perturbation='paraphrasing', aggregation='rougeL', n_perturb=5)

        question_1 = 'Will Jay-Z reach the age of 60 before Kendrick Lamar?'
        messages_1 = [{'role': 'user', 'content': question_1}]
        ret_1 = spuq.run(messages_1, temperature=0.)
        confidence_1 = ret_1['confidence']
        print('case 1', ret_1)

        question_2 = 'Is 100 greater than 3?'
        messages_2 = [{'role': 'user', 'content': question_2}]
        ret_2 = spuq.run(messages_2, temperature=0.)
        confidence_2 = ret_2['confidence']
        print('case 2', ret_2)

        self.assertTrue(confidence_1 < confidence_2)


if __name__ == '__main__':
    unittest.main()
