
# Grantland NCAA March Madness Knockout Pool Predictor
This repo contains a class for selecting "optimal" predictions for the Grantland NCAA March Madness Knockout Pool: http://grantland.com/the-triangle/the-official-grantland-march-madness-knockout-pool/

It works by performing a pruned backtracking search through all possible selections up until a specified day of the tournament, and returns the optimal selection sequence. Where "optimal" means maximizing the tournament-round probabilities over the whole of the tournament. It relies on the 538 predictions: https://projects.fivethirtyeight.com/2018-march-madness-predictions/ (the program automatically downloads the 2018 predictions)

538 doesn't seem to update the csv file containing round probabilities during the tournament, so this program really only works for the first round of the tournament unless you update the csv file. 

Also, note that the approach is not actually optimal due to some approximations. Specifically, it allows for selecting teams that are implicitly defeated by an earlier selection. It also doesn't account for the predictions of other people in the knockout pool, which you have to consider for it to really be optimal.

Finally, it's possible there's a dynamic programming solution to this problem, but I haven't been able figure one out if there is one. If you read this and find one, let me know.

## running
you need to install numpy and pandas to use this file:
```
pip install numpy pandas
```

then to run it 
```
python select_predictions.py
```

the final two lines printed give the probability of the order of selections 
along with the corresponding selection
