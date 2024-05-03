from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from evaluate import load as hf_load
import numpy as np
import pdb

class TextSimilarity:

    def __init__(self) -> None:
        self.embedder = SentenceTransformer( 'paraphrase-multilingual-mpnet-base-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bertscore = hf_load("bertscore")


    def score(self, a: str, b: str, method: str) -> float:
        
        if method == 'sbert':
            # the cosine similarity of the sentence-bert embedding
            # see https://arxiv.org/abs/1908.10084

            embs = self.embedder.encode([a, b])
            norm = np.sqrt((embs * embs).sum(-1))
            norm_embs = embs / norm.reshape(-1, 1)
            cos_sim = (norm_embs[0] * norm_embs[1]).sum(-1)
            return cos_sim
        
        elif method in ['rouge1', 'rouge2', 'rougeL']:
            # ROUGE score: https://en.wikipedia.org/wiki/ROUGE_(metric)

            return self.rouge_scorer.score(a, b)[method].fmeasure

        elif method == 'bertscore':
            # BERTScore: https://arxiv.org/abs/1904.09675
            # which cross check the contextualized word embedding between two sentence

            results = self.bertscore.compute(predictions=[b], references=[a], lang="en")
            return results['f1'][0]
        
        else:
            raise ValueError('text_similarity method not supported: %s'%method)
        