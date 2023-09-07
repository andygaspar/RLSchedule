import time

import numpy as np
import pandas as pd
from data_maker import make_cost_vectors
import matplotlib.pyplot as plt

df = pd.read_csv('flights.csv')

t = time.time()
costs = make_cost_vectors(df)
print(time.time() - t)

new_cols = pd.DataFrame(costs)
new_df = pd.concat([df, new_cols], axis=1)
new_df.to_csv('eham_flights_and_costs.csv', index_label=False, index=False)

