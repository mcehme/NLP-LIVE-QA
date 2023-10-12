from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
import os
class BM25():
    def __init__(self, dir):
        self.dir = dir
        dicts = list()
        for filename in os.listdir(dir):
            file = os.path.join(dir, filename)
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    document = {'content':text, 'meta':{'name':filename}}
                dicts.append(document)
        self.docstore = InMemoryDocumentStore(use_bm25=True)
        self.docstore.write_documents(dicts)
        self.retriever = BM25Retriever(self.docstore)

    def topK(self, k, query):
        return self.retriever.retrieve(query, top_k=k)