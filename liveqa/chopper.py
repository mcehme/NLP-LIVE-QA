from transformers import AutoTokenizer
class Chopper():
    
    def __init__(self, max_passage, safety=5):
        self.max_passage = max_passage
        self.tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
        self.safety = safety

    def chop(self, doc, query):
        query_size = len(self.tokenizer.tokenize(query))
        text = doc.to_dict()['content']
        text_tokens = self.tokenizer.tokenize(text)
        n = self.max_passage - query_size - self.safety
        
        passages = [self.tokenizer.convert_tokens_to_string(text_tokens[i:i+n]) for i in range(0, len(text_tokens), n)]
        return passages

    def chopAll(self, docs, query):
        passages = list()
        for doc in docs:
            passages.extend(self.chop(doc, query))
        return passages

    # Load model directly

