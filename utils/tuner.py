from liveqa import pipe
import json
import os
import re
from numpy import arange

class Tuner():
    def __init__(self, output_dir, k= 10, data_dir='./data/', output_basename='tuner'):
        if os.path.exists(output_dir) and not os.path.isdir(output_dir):
            raise Exception('Output_dir must be a directory.')
        self.output_dir = output_dir
        self.pipeline = pipe.QAPipeline(k, 1, data_dir)
        self.output_basename = output_basename

    def tune(self, q_file, start, stop, step):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        with open(q_file, 'r') as f:
            questions = json.load(f)
        results = dict()
        for i in arange(start, stop, step):
            self.pipeline.set_threshold(i)
            result = self.pipeline.batch_execute(questions)
            results[f'Threshold: {i}'] = result
        return self.__write_results(results)
         
    
    def __write_results(self, results):
        r = re.compile(f'{self.output_basename}([0-9]+)')
        dir_contents = os.listdir(self.output_dir)
        
        # manually doing this because we can combine filtering, mapping, and max checking into 1 loop
        maximum = -1
        for file in dir_contents:
            number = r.search(file).group(1)
            if number is not None and int(number) > maximum:
                maximum = int(number)
        with open(f'{self.output_dir}{self.output_basename}{maximum+1}.json', 'w') as f:
            json.dump(results, f)
        return f'{self.output_dir}{self.output_basename}{maximum+1}.json'

        


                
        




        
        


    
