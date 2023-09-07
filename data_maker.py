import multiprocessing
from multiprocessing import Pool

import numpy as np
import pandas as pd
from CostPackage.arrival_costs import get_cost_model, get_data_dict

from curfew import get_curfew_threshold
from curfew import get_flight_length

df_airline = pd.read_csv("Hotspot_Dataframes/airport_airline_frequency.csv")
aircraft_seats = get_data_dict()["aircraft_seats"]
df_capacity = pd.read_csv("Hotspot_Dataframes/airport_max_capacity.csv")
pax = pd.read_csv("Hotspot_Dataframes/pax.csv")

# iata increase load factor
pax.pax = pax.pax.apply(lambda pp: int(pp + pp * 0.021))
df_turnaround = pd.read_csv('Hotspot_Dataframes/turnaround.csv')
df_airport_airline_aircraft = \
    pd.read_csv("Hotspot_Dataframes/airport_airline_cluster_frequency.csv")

turnaround_dict = dict(zip(df_turnaround.AirCluster, df_turnaround.MinTurnaround))

regulations = pd.read_csv("Hotspot_Dataframes/regulations_25_nonzero.csv")


def get_cost_vect_parallel(args_tuple):
    line, load_factor, times, seed = args_tuple
    airline = line.Company
    airport = line.Destination
    fl_type = line.aircraft_cluster

    passengers = get_passengers(airport=airport, airline=airline,
                                air_cluster=fl_type, load_factor=load_factor, seed=seed)
    missed_connected = get_missed_connected(airport=airport, airline=airline, passengers=passengers, seed=seed)
    length = get_flight_length(airline=airline, airport=airport, air_cluster=fl_type, seed=seed)

    eta = line.new_time
    min_turnaround = turnaround_dict[fl_type]
    curfew_th, rotation_destination = get_curfew_threshold(airport, airline, fl_type, eta, min_turnaround, seed=seed)
    curfew = (curfew_th, get_passengers(rotation_destination, airline, fl_type, load_factor, seed=seed)) \
        if curfew_th is not None else None

    low_cost = line.is_low
    cost_fun = get_cost_model(aircraft_type=fl_type, is_low_cost=low_cost, destination=airport,
                              length=length, n_passengers=passengers, missed_connected=missed_connected,
                              curfew=curfew)

    delay_cost_vect = np.array([cost_fun(j) for j in range(300)])

    return delay_cost_vect


def make_cost_vectors(df):
    load_factor = 0.89
    times = np.linspace(0, 300, 300)
    seed = 0

    with Pool(multiprocessing.cpu_count()) as p:
        args = [(line, load_factor, times, seed) for i, line in df.iterrows()]
        results = p.map(get_cost_vect_parallel, args)

    return np.array(results)


def get_passengers(airport, airline, air_cluster, load_factor, seed=None):
    pax_local = pax[(pax.destination == airport)
                    & (pax.airline == airline)
                    & (pax.air_cluster == air_cluster)]
    if pax_local.shape[0] > 0:
        flight_sample = pax_local.leg1.sample(random_state=seed).iloc[0]
        passengers = pax_local[pax_local.leg1 == flight_sample].pax.sum()
    else:
        passengers = int(aircraft_seats[aircraft_seats.Aircraft == air_cluster]["SeatsLow"].iloc[0]
                         * load_factor)
    return passengers


def get_missed_connected(airport, airline, passengers, seed=None):
    pax_connections = pax[(pax.destination == airport) & (pax.airline == airline)]
    if pax_connections.shape[0] > 0:
        pax_connections = pax_connections.sample(n=passengers, weights=pax_connections.pax, replace=True,
                                                 random_state=seed)
        pax_connections = pax_connections[pax_connections.leg2 > 0]
        if pax_connections.shape[0] > 0:
            missed_connected = pax_connections.apply(lambda x: (x.delta_leg1, x.delay), axis=1).to_list()
        else:
            missed_connected = None
    else:
        missed_connected = None

    return missed_connected
