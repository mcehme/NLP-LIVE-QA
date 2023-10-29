from liveqa import qa, reranker, retriever, chopper
import time
class QAPipeline():
    def __init__(self, k, threshold, data_dir):
        self.bm25 = retriever.BM25(data_dir)
        
        self.qabert = qa.QABert(0.01)
        self.choppy = chopper.Chopper(512)
        self.k = k
        self.threshold = .01
    def execute(self, query):
        docs = self.bm25.topK(self.k, query)
        passages = self.choppy.chopAll(docs, query)

        ans = None
        answers_scores = []
        for passage in passages:
            answer, score = self.qabert.evaluatePassage(passage, query)
            if score>= self.threshold:
                answers_scores.append((answer,score))
        # For debugging or inspection, print the answers and their scores.
        print("All answers and scores:")
        for idx, (answer, score) in enumerate(answers_scores):
            print(f"Answer {idx + 1}: {answer}, Score: {score}")

        if answers_scores:
            best_answer = max(answers_scores, key=lambda item:item[1])[0]
        else:
        # Handle the case when no suitable answer was found.
            best_answer = None
        return best_answer is not None, best_answer


    def batch_execute(self, queries):
        results = dict()
        for query in queries:
            start = time.perf_counter_ns()
            success, ans = self.execute(query)
            end = time.perf_counter_ns()
            results[query] = {'success':success, 'answer':ans, 'time': end - start}
        return results
    def get_threshold(self):
        return self.threshold
    def set_threshold(self, threshold):
        self.threshold = threshold
    