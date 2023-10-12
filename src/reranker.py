from transformers import AutoModelForSequenceClassification,AutoTokenizer, pipeline

class BertReRanker():
    def __init__(self, threshold):
        self.threshold = threshold
        model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
        tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
        self.classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
    

    def pickAcceptable(self, passages, query):
        for passage in passages:
            args = {'text':query, 'text_pair':passage}
            score = self.classifier(args)['score']
            if score >= self.threshold:
                return passage
        return None
    def pickBest(self, passages, query):
        best = None
        bestScore = None
        for passage in passages:
            args = {'text':passage, 'text_pair':query}
            score = self.classifier(args)['score']
            if best is None or score > bestScore:
                print(score)
                best = passage
                bestScore = score
        return best



bert = BertReRanker(0.8)
passages = [r'''Tokyo (/ˈtoʊkioʊ/;[7] Japanese: 東京, Tōkyō, [toːkʲoː] ⓘ), officially the Tokyo Metropolis (東京都, Tōkyō-to), is the capital and the most populous prefecture of Japan.[8] Tokyo's metropolitan area (including neighboring prefectures as well as Tochigi, Gunma and Ibaraki; 13,452 square kilometers or 5,194 square miles) is the most populous in the world, with an estimated 37.468 million residents as of 2018;[9] although this number has been gradually decreasing since then, the prefecture itself has a population of 14.09 million people[4] while the prefecture's central 23 special wards have a population of 9.73 million.[10] Located at the head of Tokyo Bay, the prefecture forms part of the Kantō region on the central coast of Honshu, Japan's largest island. Tokyo serves as Japan's economic center and is the seat of both the Japanese government and the Emperor of Japan.''', r'''Atlanta (/ætˈlæntə/ at-LAN-tə, or /ætˈlænə/ at-LAN-ə) is the capital and most populous city of the U.S. state of Georgia. It is the seat of Fulton County, although a portion of the city extends into neighboring DeKalb County. With a population of 498,715 living within the city limits, it is the eighth most populous city in the Southeast and 38th most populous city in the United States according to the 2020 U.S. census.[9] It is the core of the much larger Atlanta metropolitan area, which is home to nearly 7 million people, making it the eighth-largest metropolitan area in the United States.[11] Situated among the foothills of the Appalachian Mountains at an elevation of just over 1,000 feet (300 m) above sea level, it features unique topography that includes rolling hills, lush greenery, and the most dense urban tree coverage of any major city in the United States.[13]''']
query = 'What is the capital of Japan'
result = bert.pickAcceptable(passages, query)
print(result)
result = bert.pickBest(passages, query)
print(result)
