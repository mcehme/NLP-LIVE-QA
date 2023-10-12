from transformers import pipeline

class BertReRanker():
    def __init__(self, threshold):
        self.threshold = threshold
        self.classifier = pipeline('text-classification', model='amberoad/bert-multilingual-passage-reranking-msmarco')

    

    def pickAcceptable(self, passages, query):
        for passage in passages:
            args = {'text':query, 'text_pair':passage}
            result = self.classifier(args)
            if result['label'] == 'LABEL_1' and result['score'] >= self.threshold:
                return True, passage
        return False, None
    def pickBest(self, passages, query):
        best = None
        bestScore = None
        for passage in passages:
            args = {'text':query, 'text_pair':passage}
            result = self.classifier(args)
            if best is None or result['label'] == 'LABEL_1' and result['score'] >= bestScore:
                best = passage
                bestScore = result['score'] if result['label'] == 'LABEL_1' else -1
        return best