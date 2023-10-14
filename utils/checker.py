import json
class Checker():
    def check(self, answer_file):
        with open(answer_file, 'r') as f:
            answers = json.load(f)
        results = self.__organize_answers(answers)

        valid_answers = list()
        invalid_answers = list()
        stats = dict()
      
        for question in results:
            for threshold in results[question]:
                stats[threshold] = stats.get(threshold, dict())
                stats[threshold]['AT'] = stats.get('AT', 0) + results[question][threshold]['time']
                if results[question][threshold]['answer'] in valid_answers:
                    if results[question][threshold]['success'] is True:
                        stats[threshold]['TP'] = stats[threshold].get('TP', 0) + 1
                    else: 
                        stats[threshold]['TN'] = stats[threshold].get('TN', 0) + 1
                    continue
                if results[question][threshold]['answer'] in invalid_answers:
                    if results[question][threshold]['success'] is True:
                        stats[threshold] = stats.get(threshold, dict())
                        stats[threshold]['FP'] = stats[threshold].get('FP', 0) + 1
                    else: 
                        stats[threshold] = stats.get(threshold, dict())
                        stats[threshold]['FN'] = stats[threshold].get('FN', 0) + 1
                    continue
                print(f'Question: {question}')
                if results[question][threshold]["success"]:
                    print(f'Answer: {results[question][threshold]["answer"]}')
                else:
                    print('UNANSWERABLE')
                print('Is the above response correct given the corpus?')
                user = input('Y/n:\t').upper()
                if user == 'Y':
                    if results[question][threshold]['success'] is True:
                        stats[threshold]['TP'] = stats[threshold].get('TP', 0) + 1
                    else: 
                        stats[threshold]['TN'] = stats[threshold].get('TN', 0) + 1
                    valid_answers.append(results[question][threshold]['answer'])
                else:
                    if results[question][threshold]['success'] is True:
                        stats[threshold] = stats.get(threshold, dict())
                        stats[threshold]['FP'] = stats[threshold].get('FP', 0) + 1
                    else: 
                        stats[threshold] = stats.get(threshold, dict())
                        stats[threshold]['FN'] = stats[threshold].get('FN', 0) + 1
                    invalid_answers.append(results[question][threshold]['answer'])
            valid_answers = list()
            invalid_answers = list()
        for threshold in stats:
            stats[threshold]['AT'] = stats[threshold]['AT']/len(results)
        return stats

    def __organize_answers(self, answers):
        results = dict()
        for threshold in answers:
            for question in answers[threshold]:
                results[question] = results.get(question, dict())
                results[question][threshold] = answers[threshold][question]
        return results


