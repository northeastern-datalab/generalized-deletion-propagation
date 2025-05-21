import csv
import os
import platform

import numpy as np
import pandas as pd

from datetime import datetime
from src.constants import queries
from src.utils import addTuples, computeWitnesses
from src.specialized_algorithms import smallest_witness_problem
from src.gdp import gdp
from src.query import Query



def runTestCase():
    CREATE_DATA = True
    OVERWRITE_DATA = True
    DATA_SAVE_FILE = 'data/expt_output/expt-data-case-study.csv'
    DATA_INSTANCE_PICKLE_FOLDER = 'data/data_instances_pickle_dump/'
    EXPT_CASE_DETAILS_FILE = 'expt/synthetic/synthetic_gdp_data_expt_cases.csv'

    # Delete k% of expenses
    k = 2
    query_del = [Query('airport-fee', ['AIRPORT', 'COST'], [('Airport', ['AIRPORT','COST']),('Flight', ['OP_CARRIER', 'AIRPORT', 'DEST', 'COST', 'IS_POPULAR'] ) ]),
                 Query('airport-fee', ['AIRPORT', 'COST'], [('Airport', ['AIRPORT','COST']),('Flight', ['OP_CARRIER', 'ORIGIN', 'AIRPORT', 'COST', 'IS_POPULAR'] ) ]),
                 Query('flight-fee', ['OP_CARRIER', 'ORIGIN', 'DEST', 'COST', 'IS_POPULAR'], [('Flight', ['OP_CARRIER', 'ORIGIN', 'DEST', 'COST', 'IS_POPULAR'] )]), 
                ]                  
        
    # Minimize deletion of connected locations
    query_min = [Query('0-hop', ['ORIGIN', 'DEST'], [('Flight', ['OP_CARRIER', 'ORIGIN', 'DEST', 'COST3', 'IS_POPULAR1']), ('Airport', ['ORIGIN', 'COST1']), 
                ('Airport', ['DEST', 'COST2'])]),
                Query('1-hop', ['ORIGIN', 'DEST'], [('Flight', ['OP_CARRIER', 'ORIGIN', 'C1', 'COST4', 'IS_POPULAR1']), ('Flight', ['OP_CARRIER', 'C1', 'DEST', 'COST5','IS_POPULAR2']), ('Airport', ['ORIGIN', 'COST1']), ('Airport', ['C1', 'COST2']), ('Airport', ['DEST', 'COST3'])]),
                # Query('2-hop', ['ORIGIN', 'DEST'], [('Flight', ['OP_CARRIER', 'ORIGIN', 'C1', 'COST5', 'IS_POPULAR1']), ('Flight', ['OP_CARRIER', 'C1', 'C2', 'COST6', 'IS_POPULAR2']), ('Flight', ['OP_CARRIER', 'C2', 'DEST', 'COST7', 'IS_POPULAR3']),('Airport', ['ORIGIN', 'COST1']), ('Airport', ['C1', 'COST2']), ('Airport', ['C2', 'COST3']), ('Airport', ['DEST', 'COST4'])])
    ]
    
    query_pres = [Query('0-hop', ['ORIGIN', 'DEST'], [('Flight', ['OP_CARRIER', 'ORIGIN', 'DEST', 'COST3', 'IS_POPULAR1']), ('Airport', ['ORIGIN', 'COST1']), 
                ('Airport', ['DEST', 'COST2'])],  constants={'IS_POPULAR1': 1}),
                Query('1-hop', ['ORIGIN', 'DEST'], [('Flight', ['OP_CARRIER', 'ORIGIN', 'C1', 'COST4', 'IS_POPULAR1']), ('Flight', ['OP_CARRIER', 'C1', 'DEST', 'COST5','IS_POPULAR2']), ('Airport', ['ORIGIN', 'COST1']), ('Airport', ['C1', 'COST2']), ('Airport', ['DEST', 'COST3'])],  constants={'IS_POPULAR1': 1, 'IS_POPULAR2': 1}),
                # Query('2-hop', ['ORIGIN', 'DEST'], [('Flight', ['OP_CARRIER', 'ORIGIN', 'C1', 'COST5', 'IS_POPULAR1']), ('Flight', ['OP_CARRIER', 'C1', 'C2', 'COST6', 'IS_POPULAR2']), ('Flight', ['OP_CARRIER', 'C2', 'DEST', 'COST7', 'IS_POPULAR3']),('Airport', ['ORIGIN', 'COST1']), ('Airport', ['C1', 'COST2']), ('Airport', ['C2', 'COST3']), ('Airport', ['DEST', 'COST4'])], constants={'IS_POPULAR1': 1, 'IS_POPULAR2': 1, 'IS_POPULAR3': 1}) 
    ]

    run_id = datetime.now().timestamp()
    database_instance = None
    tuple_weights = None

    flight_data = pd.read_csv('data/case_study/flights.csv')
    airline_data = pd.read_csv('data/case_study/airlines.csv')
    airport_data = pd.read_csv('data/case_study/airports.csv')

    tuple_weights = {}

    for _, row in flight_data.iterrows():
        tuple_weights['Flight_'+'_'.join([str(row[v]) for v in flight_data.columns])] = row['COST']
    for _, row in airport_data.iterrows():
        tuple_weights['Airport_'+'_'.join([str(row[v]) for v in airport_data.columns])] = row['COST']

    
    airline_names = [a[1]['OP_CARRIER'] for a in airline_data.iterrows()]
    for airline in airline_names:
        airline_flight_data = flight_data[flight_data['OP_CARRIER'] == airline]

        database_instance = {'Flight': set([tuple(f) for f in airline_flight_data.values]), 
                         'Airport': set([tuple(f) for f in airport_data.values])}
        

        instance_id = datetime.now().timestamp()

        output = {}

        output['instance timestamp'] = instance_id
        output['run id'] = run_id
        output['airline'] = airline
        output['processor'] = platform.processor()


        witnesses_del = [computeWitnesses(q, database_instance) for q in query_del]
        projections_del  = [witnesses_del[i][q.head_vars] for i, q in enumerate(query_del)]
        kdel = int((sum([len(projections) for projections in projections_del]) * k)/ 100)

        witnesses_pres = [computeWitnesses(q, database_instance) for q in query_pres]
        projections_pres  = [witnesses_pres[i][q.head_vars] for i,q in enumerate(query_pres)]
        kpres = [len(projections) for projections in projections_pres]
        

        witnesses_min = [computeWitnesses(q, database_instance) for q in query_min]
        projections_min  = [witnesses_min[i][q.head_vars] for i,q in enumerate(query_min)]
        output['number_of_witnesses'] = [len(projections) for projections in projections_min]

        expt_results = {}

        output['kpres'] = kpres
        output['kdel'] = kdel
        print([len(projections) for projections in projections_del])

        expt_results["lp_results"] = gdp(database_instance, query_del = query_del, query_pres = query_pres, kdel=kdel, kpres=kpres, query_max = None, query_min = query_min, tuple_weights = tuple_weights, lp_type = "LP", kdel_is_total = True, treat_views_as_unions=True, allow_empty = True)
        
        expt_results["ilp_results"] = gdp(database_instance, query_del = query_del, query_pres = query_pres, kdel=kdel, kpres=kpres, query_max = None, query_min = query_min, tuple_weights = tuple_weights, lp_type = "ILP", kdel_is_total = True, treat_views_as_unions=True, allow_empty = True)

        expt_results["ilp_results_60"] = gdp(database_instance, query_del = query_del, query_pres = query_pres, kdel=kdel, kpres=kpres, query_max = None, query_min = query_min, tuple_weights = tuple_weights, lp_type = "ILP", kdel_is_total = True, treat_views_as_unions=True, time_limit=60, allow_empty = True)

        for expt_name in expt_results:
            output.update({expt_name+': ' + str(key): val for key, val in expt_results[expt_name].items()})

        
        if not os.path.exists(DATA_SAVE_FILE) or OVERWRITE_DATA:
            
            with open(DATA_SAVE_FILE, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=output.keys())
                writer.writeheader()
                writer.writerow(output)
            OVERWRITE_DATA = False
        else:            
            
            with open(DATA_SAVE_FILE, 'a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=output.keys())
                writer.writerow(output)


if __name__ == '__main__':
    runTestCase()
