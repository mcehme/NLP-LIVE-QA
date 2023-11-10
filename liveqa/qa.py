from transformers import pipeline
class QABert:
    '''Question Answering Transformer using Bert
    
    :param threshold: the threshold in range (0, 1) telling us how "good" answers must be. Should be relatively small (< 0.1)'''

    def __init__(self, threshold):
        self.pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.threshold = threshold

    def runQuery(self, passage, query):
        '''Given a passage and a query, attempts to answer the query
        
        :param passage: the passage to use as context
        :param query: the question to answer
        :rtype: tuple
        :return: a 2-tuple with a boolean success value and either the answer or None'''
        result = self.pipe(question=query, context=passage)
        score = result['score']
        if score >= self.threshold:
            return True, result['answer']
        return False, None
    def evaluatePassage(self, passage, query):
        result = self.pipe(question=query, context=passage)
        return result['answer'], result['score']
