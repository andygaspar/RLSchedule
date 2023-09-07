import numpy as np
import pandas as pd
import gurobipy as gb

df_flights = pd.read_csv('Hotspot_Dataframes/flights_complete.csv')
df_f = df_flights[df_flights.Destination == 'EHAM'].copy(deep=True)
df_f.sort_values(by=['arr_day', 'arr_min'], inplace=True)
df_f['new_time'] = -1


days = df_f.arr_day.unique()

for day in days:
    m = gb.Model()
    df_day = df_f[df_f.arr_day == day]

    x = m.addMVar((df_day.shape[0], 1440), vtype=gb.GRB.BINARY)
    z = m.addMVar(df_day.shape[0], vtype=gb.GRB.INTEGER, ub=50)
    m.addConstr(x.sum(axis=1) == 1)
    # m.addConstr(x.sum(axis=0) <= 1)

    arr_time = np.repeat(np.expand_dims(df_day.arr_min.to_numpy(), axis=1), x.shape[1], axis=1)
    new_arr_time = np.repeat(np.expand_dims(range(1440), axis=0), df_day.shape[0], axis=0)

    m.addConstr((x * (arr_time - new_arr_time)).sum(axis=1) <= z)
    m.addConstr((x * (arr_time - new_arr_time)).sum(axis=1) >= -z)

    m.addConstr(
        (x[:-1] * (new_arr_time[:-1]) - x[1:] * (new_arr_time[1:])).sum(axis=1)
        <= -1)

    m.setObjective(z.sum())

    m.optimize()
    df_f.loc[df_f.arr_day == day, 'new_time'] = (x.x * new_arr_time).sum(axis=1)


df_f.new_time = df_f.new_time.astype(int)
df_f.to_csv('flights.csv', index_label=False, index=False)




# (x.x*new_arr_time).sum(axis=1)
#
# df_day.arr_min
# days = df_f.arr_day.unique()
# print(z.x)
