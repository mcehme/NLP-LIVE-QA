import statistics
import math
class StatGenerator():
    def get_stats(self, stats):

        final_stats = dict()
        error_rates = list()
        latencies = list()

        for threshold in stats:
            stat = stats[threshold]
            TP = stat.get('TP', 0)
            TN = stat.get('TN', 0)
            FP = stat.get('FP', 0)
            FN = stat.get('FN', 0)
            latency = stat.get('AT', -1)
            final_stats[threshold] = {'accuracy':self.__accuracy(TP, TN, FP, FN),
                                      'error rate': self.__error_rate(TP, TN, FP, FN),
                                      'false answer rate': self.__false_answer_rate(TP, TN, FP, FN),
                                      'false unanswerable rate': self.__false_unanswerable_rate(TP, TN, FP, FN),
                                      'precision':self.__precision(TP, TN, FP, FN),
                                      'recall': self.__recall(TP, TN, FP, FN),
                                      'f1': self.__f1(TP, TN, FP, FN),
                                      'latency': latency
                                      }
            latencies.append(latency)
            error_rates.append(self.__error_rate(TP, TN, FP, FN))
        avg_latency = statistics.mean(latencies)
        avg_error_rate = statistics.mean(error_rates)

        std_latency = statistics.stdev(latencies)
        std_error_rate = statistics.stdev(error_rates)

        z_latency = lambda x: (x - avg_latency)/std_latency
        z_error_rate = lambda x: (x - avg_error_rate)/(std_error_rate if std_error_rate != 0 else 1)

        min_value = 0
        min_threshold = None

        for threshold in stats:
            stat = stats[threshold]
            TP = stat.get('TP', 0)
            TN = stat.get('TN', 0)
            FP = stat.get('FP', 0)
            FN = stat.get('FN', 0)
            latency = stat.get('AT', -1)
            error_rate = self.__error_rate(TP, TN, FP, FN)

            latency_standard = z_latency(latency)
            error_rate_standard = z_error_rate(error_rate)

            weighted_score = latency_standard + error_rate_standard

            final_stats[threshold]['weighted score'] = weighted_score
            if min_threshold is None or weighted_score < min_value:
                min_value = weighted_score
                min_threshold = threshold
 
        return final_stats, min_threshold
    

    def get_stats_baseline(self, stats):
        TP = stats.get('TP', 0)
        TN = stats.get('TN', 0)
        FP = stats.get('FP', 0)
        FN = stats.get('FN', 0)
        latency = stats.get('AT', -1)

        final_stats = {'accuracy':self.__accuracy(TP, TN, FP, FN),
                        'error rate': self.__error_rate(TP, TN, FP, FN),
                        'false answer rate': self.__false_answer_rate(TP, TN, FP, FN),
                        'false unanswerable rate': self.__false_unanswerable_rate(TP, TN, FP, FN),
                        'precision':self.__precision(TP, TN, FP, FN),
                        'recall': self.__recall(TP, TN, FP, FN),
                        'f1': self.__f1(TP, TN, FP, FN),
                        'latency': latency
                        }
        return final_stats


    def __accuracy(self, TP, TN, FP, FN):
        return (TP + TN)/(TP + TN + FP + FN)
    def __error_rate(self, TP, TN, FP, FN):
        return 1 - self.__accuracy(TP, TN, FP, FN)
    def __false_answer_rate(self, TP, TN, FP, FN):
        return FP/(TP + TN + FP + FN)
    def __false_unanswerable_rate(self, TP, TN, FP, FN):
        return FN/(TP + TN + FP + FN)
    def __precision(self, TP, TN, FP, FN):
        return TP/(TP + FP)
    def __recall(self, TP, TN, FP, FN):
        return TP/(TP + FN)
    def __f1(self, TP, TN, FP, FN):
        return 2*self.__precision(TP,TN,FP, FN)*self.__recall(TP, TN, FP, FN)/(self.__precision(TP,TN,FP, FN)+self.__recall(TP, TN, FP, FN))
    


        