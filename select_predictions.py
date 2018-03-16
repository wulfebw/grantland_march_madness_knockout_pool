
import copy
import numpy as np
import os
import pandas as pd
import urllib.request

# round 2 actually corresponds to first day of tournament
DAY2ROUND = {
    0:2,
    1:2,
    2:3,
    3:3,
    4:4,
    5:4,
    6:5,
    7:5,
    8:6,
    9:7
}

''' data loading utilities '''

def maybe_download(filepath, url):
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)  

def team_names_for_query(data, query):
    return list(data.query(query)['team_name'])

def extract_days(data, n_days=10):
    days = {i:[] for i in range(n_days)}

    # 0=15, 1=16
    seeds = "['5','12','4','13','6','11','3','14']"
    query = "(team_region == 'South' or team_region == 'West') and team_seed in {}".format(seeds)
    days[0] += team_names_for_query(data, query)
    query = "(team_region == 'Midwest') and team_seed in {}".format(seeds)
    days[1] += team_names_for_query(data, query)
    seeds = "['1','16','8','9','6','11','3','14']"
    query = "(team_region == 'East') and team_seed in {}".format(seeds)
    days[0] += team_names_for_query(data, query)

    seeds = "['1','16','8','9','7','10','2','15']"
    query = "(team_region == 'South' or team_region == 'West') and team_seed in {}".format(seeds)
    days[1] += team_names_for_query(data, query)
    query = "(team_region == 'Midwest') and team_seed in {}".format(seeds)
    days[0] += team_names_for_query(data, query)
    seeds = "['1','16','8','9','6','11','3','14']"
    query = "(team_region == 'East') and team_seed not in {}".format(seeds)
    days[1] += team_names_for_query(data, query)
    
    # 2=17, 3=18
    seeds = "['5','12','4','13','6','11','3','14']"
    query = "(team_region == 'South' or team_region == 'West') and team_seed in {}".format(seeds)
    days[2] += team_names_for_query(data, query)
    query = "(team_region == 'Midwest') and team_seed in {}".format(seeds)
    days[3] += team_names_for_query(data, query)
    seeds = "['1','16','8','9','7','10','2','15']"
    query = "(team_region == 'South' or team_region == 'West') and team_seed in {}".format(seeds)
    days[3] += team_names_for_query(data, query)
    query = "(team_region == 'Midwest') and team_seed in {}".format(seeds)
    days[2] += team_names_for_query(data, query)
    seeds = "['1','16','8','9','6','11','3','14']"
    query = "(team_region == 'East') and team_seed in {}".format(seeds)
    days[2] += team_names_for_query(data, query)
    query = "(team_region == 'East') and team_seed not in {}".format(seeds)
    days[3] += team_names_for_query(data, query)

    # 4=22, 5=23, 6=24, 7=25
    query = "team_region == 'South' or team_region == 'West'"
    days[4] += team_names_for_query(data, query)
    days[6] += team_names_for_query(data, query)
    query = "team_region == 'Midwest' or team_region == 'East'"
    days[5] += team_names_for_query(data, query)
    days[7] += team_names_for_query(data, query)

    # 8=31, 9= april 2nd
    days[8] += list(data['team_name'])
    days[9] += list(data['team_name'])
    
    # convert to sets
    for i in range(n_days):
        days[i] = set(days[i])

    return days

def load_data(filepath, min_rd2_win=.8, date='2018-03-13'):
    data = pd.read_csv(filepath)

    # only interested in the mens
    data = data.loc[data['gender'] == 'mens']

    # only want latest forecast
    data = data.loc[data['forecast_date'] == date]

    # threshold win rate
    data = data.loc[data['rd2_win'] > min_rd2_win]

    # extract days
    days = extract_days(data)
    
    # convert to log probs
    rd_keys = [k for k in data.keys() if 'rd' in k]
    for rd in rd_keys:
        data[rd] = np.log(data[rd])

    return data, days

''' team selector '''

def get_valid_names(days, day, selection):
    return list(days[day].difference(set(selection)))

def get_max_probs(data, days):
    '''
    these are used to prune the backtracking search to only potentially optimal
    selections

    this is actually an approximation of the max_probs
    and potentially incorrect at the cost of saving computation
    the reason being that the order in which you select and remove the 
    best team each day matters, and this ignores that

    but I'd rather have a ε-correct solution fast than an optimal solution 
    really slowly in this case because there's already a ton of approximation 
    built into the prediction probabilities
    '''
    max_probs = []
    selected = set()
    for day in reversed(days):
        rd_key = 'rd{}_win'.format(DAY2ROUND[day])
        best = -np.inf
        for index, row in data.iterrows():
            if row[rd_key] > best and row['team_name'] not in selected:
                selected.add(row['team_name'])
                best = row[rd_key]
        max_probs.append(best)
    max_probs = np.cumsum(max_probs[::-1])[::-1] + .3 # sick
    return list(max_probs)

class PredictionSelector(object):
    '''
    Class that performs the backtracking search
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.best_value = -np.inf
        self.best_selection = None
        self.value_name_day_cache = {}

    def select(self, data, days, max_day=None):
        if max_day is None:
            max_day = np.max(list(days.keys()))
        self.max_day = max_day
        self.max_probs = get_max_probs(data, range(max_day+1)) + [0]

        def recurse(day, value, selection):
            if value + self.max_probs[day] < self.best_value:
                return
            
            # completed the tournament up to the max day, check if better and backtrack
            if day == self.max_day + 1:

                if value > self.best_value:
                    self.best_value = value
                    self.best_selection = copy.deepcopy(selection)
                    print('\nbest_value: {}'.format(np.exp(self.best_value)))
                    print('best_selection: {}'.format(self.best_selection))

            # first two days, select a pair of teams
            elif day <= 1:

                valid_names = get_valid_names(days, day, selection)
                for i, name_1 in enumerate(valid_names):
                    value_1 = self._value_for_name_day(data, name_1, day)
                    for name_2 in valid_names[i+1:]:
                        recurse(
                            day + 1, 
                            value + value_1 + self._value_for_name_day(data, name_2, day), 
                            selection + [name_1, name_2]
                        )

            # 2nd day or later, only make a single pick
            else:

                valid_names = get_valid_names(days, day, selection)
                for name in valid_names:
                    recurse(
                        day + 1, 
                        value + self._value_for_name_day(data, name, day), 
                        selection + [name]
                    )

        recurse(0, 0, [])
        return self.best_value, self.best_selection

    def _value_for_name_day(self, data, name, day):
        if (name, day) in self.value_name_day_cache.keys():
            return self.value_name_day_cache[(name, day)]
        else:
            rd_key = 'rd{}_win'.format(DAY2ROUND[day])
            value = float(data.loc[data['team_name'] == name][rd_key])
            self.value_name_day_cache[(name, day)] = value
            return value

if __name__ == '__main__':
    filepath = 'fivethirtyeight_ncaa_forecasts.csv'
    url = 'https://projects.fivethirtyeight.com/march-madness-api/2018/fivethirtyeight_ncaa_forecasts.csv'
    maybe_download(filepath, url)
    data, days = load_data(filepath)
    selector = PredictionSelector()
    best_value, best_selection = selector.select(data, days, max_day=9)