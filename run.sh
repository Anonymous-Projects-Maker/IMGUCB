#!/bin/sh
echo 'start running'

for (( i=0; i<1; i++ ))
do
    {
        echo ${i} 'running'
        # linear
        # nohup python IMBandit.py -imgucb -imfb -imlinucb -egreedy -cucb -dataset NetHEPT  > lin${i}.txt &
        # nohup python IMBandit.py -imgucb -imfb -seed 300 -p linear -repeat 4 -resdir './SimulationResults/linear_1440' -dataset Flickr > ${i}.txt &
        # nonlinear
        nohup python IMBandit.py -imgucb -imfb -imlinucb -egreedy -cucb -nlin -dataset NetHEPT  -resdir './SimulationResults/nlin' > nlin${i}.txt &
        # nohup python IMBandit.py -imlinucb -egreedy -ucb1 -seed 300 -p nonlinear  -repeat 1 -resdir './SimulationResults/nonlinear_1557' > ${i}.txt &
    }
    sleep 3
done


# wait
#start=`date +"%s"`
#end=`date +"%s"`
#echo "time: " `expr $end - $start`

# for (( i=20; i<40; i++ ))
# do
#     {
#         echo ${i}
#         nohup python IMBandit.py -imlinucb -egreedy -ucb1 -seed 300 -p linear  -repeat 1 -resdir './SimulationResults/linear_1440' > ${i}.txt &
#     }
#     sleep 1
# done