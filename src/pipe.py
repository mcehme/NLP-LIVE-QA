import qa, reranker, retriever, chopper
class QAPipeline():
    def __init__(self, k, threshold):
        self.bm25 = retriever.BM25Retriever()
        self.rerank = reranker.BertReRanker(threshold)
        self.qabert = qa.QABert(0.01)
        self.choppy = chopper.Chopper(512)
        self.k = k
    def execute(self, query):
        docs = self.bm25.topK(self.k)
        passages = self.choppy.chopAll(docs, query)
        success, acceptable = self.rerank.pickAcceptable(query)
        ans = None
        if success:
            success, ans = self.qabert.runQuery(acceptable)
        return success, ans
    