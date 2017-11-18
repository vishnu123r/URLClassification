import pandas as pd

data = pd.read_table('classification.tsv', error_bad_lines = False)
data = data.loc[:,['Primary Category', 'URL']]

lab = set(data['Primary Category'].values)
lab = dict(enumerate(lab,1))
lab = dict (zip(lab.values(),lab.keys()))


label = list(map(lab.get, list(data['Primary Category'].values)))

data['label'] = pd.Series(label).values