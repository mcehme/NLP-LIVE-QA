from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document
class BM25():

    def __init__(self, passages):
        self.passages = passages
        self.docstore = InMemoryDocumentStore(use_bm25=True)
        documents = list()
        count = 0
        for passage in passages:
            documents.append(Document(content=passage, id=count))
            count += 1
        self.docstore.write_documents(documents=documents)
        self.retriever = BM25Retriever(self.docstore)

    def topK(self, k, query):
        return self.retriever.retrieve(query, top_k= int(len(self.passages)/ 2))