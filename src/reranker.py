from transformers import AutoModelForSequenceClassification,AutoTokenizer, pipeline

class BertReRanker():
    def __init__(self, threshold):
        self.threshold = threshold
        self.model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
        self.tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
    

    def pickAcceptable(self, passages, query):
        tokens = self.tokenizer.tokenize(query)
        for passage in passages:
            passage_tokens = self.tokenizer.tokenize(passage)
            self.model.generate()
    def pickBest(passages, query):
        pass