from utils import tuner, checker

mytuner = tuner.Tuner('./output/', k=1)
answers = mytuner.tune('./questions/questions.json', 0.1, 0.15, 0.1)
mychecker = checker.Checker()
stats = mychecker.check(answers)

for threshold in stats:
    print(f'{threshold}: {stats[threshold]}')


