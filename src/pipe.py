import qa, reranker, retriever, chopper
class QAPipeline():
    def __init__(self, k, threshold):
        self.bm25 = retriever.BM25Retriever()
        self.rerank = reranker.BertReRanker(threshold)
        self.qabert = qa.QABert()
        self.choppy = chopper.Chopper(512)
        self.k = k
    def execute(self, query):
        docs = self.bm25.topK(self.k)
        passages = self.choppy.chopAll(docs, query)
        acceptable = self.rerank.pickAcceptable(query)
        success, ans = self.qabert.runQuery(acceptable)
        return success, ans
    