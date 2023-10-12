from transformers import AutoTokenizer
import math
class Chopper():
    def __init__(self, max_passage):
        self.max_passage = max_passage
        self.tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

    def chop(self, doc, query):
        pass


    def chopAll(self, docs, query):
        passages = list()
        for doc in docs:
            passages.extend(self.chop(doc, query))
        return passages

    # Load model directly

