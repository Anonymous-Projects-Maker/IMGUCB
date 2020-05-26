# IMBandits

IMBandit.py -- Simulator. Can run the experiments with command ```python IMBandit.py``` 

BanditAlg -- Baselines for running influence maximization problems.

Oracle/degreeDiscount.py, generalGreedy.py -- Two different oracles (IM algorithm).

conf.py -- The relative parameters used for the experiments. 

SimulationResults -- The folder saved for the results. 

usage example:
```
python IMBandit.py -imgucb -imfb -linucb -egreedy -cucb -dataset Flickr -nlin -resdir './SimulationResults/result_dir'
```

#### Acknowledgement:
This implementation references much from https://github.com/Matrix-Factorization-Bandit/IMFB-KDD2019), and we appreciate the authors deeply.