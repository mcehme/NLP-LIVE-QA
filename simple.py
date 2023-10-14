from utils import tuner 
from liveqa import qa

mytuner = tuner.Tuner('./output/', k=1)
mytuner.tune('./questions/questions.json', 0.1, 1, 0.1)
