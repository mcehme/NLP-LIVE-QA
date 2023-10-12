from transformers import AutoTokenizer
class Chopper():
    def __init__(self, max_passage):
        self.max_passage = max_passage
        self.tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

    def chop(self, doc, query):
        query_size = len(self.tokenizer.tokenize(query))
        with open(doc, 'r') as f:
            text = f.read()
            text_tokens = self.tokenizer.tokenize(text)
            n = self.max_passage - query_size
            passages = [text_tokens[i:i+n] for i in range(0, len(text_tokens), n)]
        return ' '.join(passages)


    def chopAll(self, docs, query):
        passages = list()
        for doc in docs:
            passages.extend(self.chop(doc, query))
        return passages

    # Load model directly

