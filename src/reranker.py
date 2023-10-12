from transformers import pipeline

class BertReRanker():
    '''Bert based re ranker that supports first fit and best fit strategies for passage selection
    
    :param threshold: the threshold in range (0, 1) that passages must meet to be considered valid for a given query'''

    def __init__(self, threshold):
        self.threshold = threshold
        self.classifier = pipeline('text-classification', model='amberoad/bert-multilingual-passage-reranking-msmarco')

    def pickAcceptable(self, passages, query):
        '''Picks the first acceptable passage from a list of passages
        
        :param passages: an iterable of passages. Each passage should be a string.
        :param query: The query to match the passages against. Should be a string.
        :rtype: tuple
        :return: a 2-tuple containing a boolean representing success and either an acceptable passage or None'''
        for passage in passages:
            args = {'text':query, 'text_pair':passage}
            result = self.classifier(args)
            if result['label'] == 'LABEL_1' and result['score'] >= self.threshold:
                return True, passage
        return False, None
    def pickBest(self, passages, query):
        '''Picks the best passage from a list of passages
        
        :param passages: an iterable of passages. Each passage should be a string.
        :param query: The query to match the passages against. Should be a string.
        :rtype: string
        :return the best fitting passage'''
        best = None
        bestScore = None
        for passage in passages:
            args = {'text':query, 'text_pair':passage}
            result = self.classifier(args)
            if best is None or result['label'] == 'LABEL_1' and result['score'] >= bestScore:
                best = passage
                bestScore = result['score'] if result['label'] == 'LABEL_1' else -1
        return best