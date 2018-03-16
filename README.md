
# Grantland NCAA March Madness Knockout Pool Predictor
This repo contains a class for selecting "optimal" predictions for the Grantland NCAA March Madness Knockout Pool: http://grantland.com/the-triangle/the-official-grantland-march-madness-knockout-pool/

It works by performing a pruned backtracking search through all possible selections up until a specified day of the tournament, and returns the "optimal" selection sequence. Where "optimal" means maximizing the tournament-round probabilities over the tournament. It relies on the 538 predictions: https://projects.fivethirtyeight.com/2018-march-madness-predictions/
(the program automatically downloads the 2018 predictions)

538 doesn't seem to update the csv file containing round probabilities during the tournament, so this program really only works for the first round of the tournament unless you manually update the csv file. 

Also, note that this approach is not actually optimal due to some approximations. Specifically, the below approach allows for selecting teams that are implicitly defeated by an earlier selection.

Finally, it's possible there's a dynamic programming solution to this problem, but I haven't been able figure it out if there is one. If you read this and find one, let me know.

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