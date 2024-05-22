# Project Description

## SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models

SPUQ is an LLM uncertainty calibration algorithm. It provides a confidence score for each query, for a given LLM.
Experiments show that this confidence score is correlated with the generation accuracy, and therefore provides a useful LLM response evaluation metric on-the-fly.

The details of the approach are documented in our [paper](https://arxiv.org/abs/2403.02509) published at EACL-2024 Conference.

The basic idea is to check whether an LLM provides a significantly different answer when we ask the same question in a slightly different way.
If it does, we assume the LLM is not confident in this case.
SPUQ perturbs the input (including the prompt and the temperature) to get multiple outputs, and then aggregate the outputs to obtain the final confidence score.
This allows SPUQ to address both epistemic (via perturbation) and aleatoric (via sampling) uncertainties, and it provides better calibration than some of the other existing methods.

## Example

Here's an example to set up SPUQ. You can play with it using the [notebook](/demo.ipynb). The config includes:
* The target LLM you'd like to calibrate. We use `gpt-35-turbo-v0301` as an example
* The perturbation method. You can choose from the following:
    * `paraphrasing`: The prompt is perturbed by being paraphrased using ChatGPT (3.5).
    * `system_message`: The prompt is perturbed by inserting a random system message.
    * `dummy_token`: The prompt is perturbed by inserting a random dummy token.
    * `temperature`: The temperature is perturbed with a random change.
* The aggregation method. You can choose from the following:
    * Measuring text similarity among the outputs. This can be done using the [Rouge score](https://en.wikipedia.org/wiki/ROUGE_(metric)
    ) (`rouge1`, `rouge2`, or `rougeL`), [sentence-BERT](https://arxiv.org/abs/1908.10084) embedding cosine similarity (`sbert`) or [BERT-Score](https://arxiv.org/abs/1904.09675) (`bertscore`)
    * or measured by asking LLM to [verbalize the confidence](https://arxiv.org/abs/2205.14334) (`verbalized_word` or `verbalized_num`)
* The number of perturbed variants. Usually up to 5.

```python
from spuq import SPUQ
from utils.llms import LLM
llm = LLM('gpt-35-turbo-v0301')
spuq = SPUQ(llm=llm, perturbation='paraphrasing', aggregation='rougeL', n_perturb=5)
```

### A low-confidence example

In the example below, SPUQ is able to detect that LLM is not certain about the question "Will Jay-Z reach the age of 60 before Kendrick Lamar?". 
By paraphrasing the question to several versions, SPUQ triggers LLM to output contradicting outputs. As a result, the final **confidence score is low (~0.256)**
```python
question = 'Will Jay-Z reach the age of 60 before Kendrick Lamar?'
messages = [{'role': 'user', 'content': question}]
spuq.run(messages, temperature=0.7)
```

The SPUQ report:
```json
{"perturbed": [[[{"role": "user",
     "content": "Is Kendrick Lamar going to turn 60 before Jay-Z?"}],
   0.7],
  [[{"role": "user",
     "content": "Who will turn 60 first, Jay-Z or Kendrick Lamar?"}],
   0.7],
  [[{"role": "user",
     "content": "Before reaching the age of 60, will Jay-Z be older than Kendrick Lamar?"}],
   0.7],
  [[{"role": "user", "content": "Will Kendrick Lamar hit 60 after Jay-Z?"}],
   0.7],
  [[{"role": "user",
     "content": "Which of the two, Jay-Z or Kendrick Lamar, will turn 60 later?"}],
   0.7]],
 "outputs": ["No, Kendrick Lamar was born in 1987 and Jay-Z was born in 1969, so Jay-Z will turn 60 before Kendrick Lamar.",
  "Jay-Z will turn 60 first. He was born on December 4, 1969, while Kendrick Lamar was born on June 17, 1987.",
  "Yes, Jay-Z is already older than Kendrick Lamar. Jay-Z was born on December 4, 1969, while Kendrick Lamar was born on June 17, 1987.",
  "I'm sorry, as an AI language model, I cannot predict the future.",
  "Jay-Z will turn 60 later."],
 "confidence": 0.25582140902337946}
```

### A high-confidence example

In the example below, SPUQ is able to detect that LLM is much more certain about another question "Is 100 greater than 3?".
Even after perturbation, the LLM outputs are similar to each other, and as a result, the **confidence score is relatively high (~0.538)**

```python
question = 'Is 100 greater than 3?'
messages = [{'role': 'user', 'content': question}]
spuq.run(messages, temperature=0.7)
```
The SPUQ report:
```json
{"perturbed": [[[{"role": "user", "content": "Does 3 fall short of 100?"}],
   0.7],
  [[{"role": "user", "content": "Is 3 less than 100?"}], 0.7],
  [[{"role": "user", "content": "Is the number 100 bigger than 3?"}], 0.7],
  [[{"role": "user", "content": "Would 100 be considered greater than 3?"}],
   0.7],
  [[{"role": "user", "content": "Does 100 exceed 3 in value?"}], 0.7]],
 "outputs": ["Yes, 3 is much less than 100.",
  "Yes, 3 is less than 100.",
  "Yes, 100 is bigger than 3.",
  "Yes, 100 is greater than 3.",
  "Yes, 100 exceeds 3 in value."],
 "confidence": 0.5384615384615384}
 ```

# Usage Steps

### Installation

This repo is tested with Python 3.11.
Please refer to the [requirements.txt](/requirements.txt) for details.
Or install the following packages with pip

```bash
pip install numpy     
pip install sentence_transformers 
pip install rouge-score  
pip install evaluate  
pip install bert-score 
pip install openai
```

### Usage

Please see the [notebook](/demo.ipynb) for details.

# Dataset Description

Please read our [paper](https://arxiv.org/abs/2403.02509) for details.

# How to cite

```
@inproceedings{gao2024spuq,
  title={SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models},
  author={Gao, Xiang and Zhang, Jiaxin and Mouatadid, Lalla and Das, Kamalika},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={2336--2346},
  year={2024}
}
```
