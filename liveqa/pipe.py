import qa, reranker, retriever, chopper
class QAPipeline():
    def __init__(self, k, threshold, data_dir):
        self.bm25 = retriever.BM25(data_dir)
        self.rerank = reranker.BertReRanker(threshold)
        self.qabert = qa.QABert(0.01)
        self.choppy = chopper.Chopper(512)
        self.k = k
    def execute(self, query):
        docs = self.bm25.topK(self.k, query)
        passages = self.choppy.chopAll(docs, query)
        success, acceptable = self.rerank.pickAcceptable(passages, query)
        ans = None
        if success:
            success, ans = self.qabert.runQuery(acceptable, query)
        return success, ans
    def batch_execute(self, queries):
        results = dict()
        for query in queries:
            success, ans = self.execute(query)
            results[query] = {'success':success, 'answer':ans}
        return results
    def get_threshold(self):
        return self.rerank.threshold
    def set_threshold(self, threshold):
        self.rerank.threshold = threshold
    