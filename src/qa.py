from transformers import pipeline
class QABert:

    def __init__(self, threshold):
        self.pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.threshold = threshold



    def runQuery(self, passage, query):
        result = self.pipe(question=query, context=passage)
        print(result)
        score = result['score']
        if score >= self.threshold:
            return True, result['answer']
        return False, ''
