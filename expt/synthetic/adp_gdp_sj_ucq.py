import argparse
import csv
import os
import pickle
import platform

import numpy as np
import pandas as pd

from datetime import datetime
from src.constants import queries
from src.utils import addTuples, computeWitnesses
from src.gdp import gdp
from src.query import Query

CREATE_DATA = True
OVERWRITE_DATA = False
DATA_SAVE_FILE = 'data/expt_output/expt-data-adp-gdp-sj-ucq-{}.csv'
DATA_INSTANCE_PICKLE_FOLDER = 'data/data_instances_pickle_dump/'
EXPT_CASE_DETAILS_FILE = 'expt/synthetic/synthetic_gdp_data_expt_cases.csv'
k = 50
k_in_percent = True

def runCase(case_no):
    """
     Read the details of the experiment from a file with specifications, and run it

    Args:
        case_no (int): The case number to be run
    """
    
    expt_case_details =  pd.read_csv(EXPT_CASE_DETAILS_FILE)
    case_details = expt_case_details[expt_case_details['case_no'] == case_no].iloc[0].to_dict()
    runTestCase(case_no, case_details['query_name'], case_details['domain_size'] , case_details['itr_log_base'], \
        case_details['itr_log_start'], case_details['itr_log_end'], case_details['num_itr'], bag_semantics = case_details['bag_semantics'],
        max_bag_size = case_details['max_bag_size'])


def runTestCase(case_no, query_name, domain_size, itr_log_base, itr_log_start, itr_log_end, itr_num, bag_semantics = True, max_bag_size = 10):
    """ 
    Given the test case parameters, run the experiment to compare times of all algorithms

    Args:
        case_no (int): The case number that is running
        query_name (str): query_name as defined in tst.constants
        domain_size (int): domain size of the database instance
        itr_log_base (float): the base that gets a higher exponent each iteration
        itr_log_start (float): starting exponent of itr_log_base
        itr_log_end (float): final exponent of itr_log_base
        itr_num (int): The number of iterations in this data run
        bag_semantics (bool, optional). Indicates if bag semantics or set semantics is used. Defaults to False.
        max_bag_size (bool, optional). Indicates max bag size under bag semantics. Defaults to 10.
    """
    if CREATE_DATA:
        runMonotoneRunOfExperiments(query_name, domain_size, itr_log_base, itr_log_start, itr_log_end, itr_num, bag_semantics = bag_semantics, max_bag_size = max_bag_size, output_filename=DATA_SAVE_FILE.format(case_no), overwrite_data = OVERWRITE_DATA)
        print('Experiment Data Generation complete. Data is stored in', DATA_SAVE_FILE.format(case_no))
    
    


def runMonotoneRunOfExperiments(query_name, domain_size, itr_log_base, itr_log_start, itr_log_end, itr_num, bag_semantics = False, max_bag_size = 10, output_filename='data/synthetic_expt/unknown_query_data.csv', overwrite_data = False, k=10, k_in_percent = True):
    """
    Calculates resilience over a monotonically bigger data instance and tracks details 

    Args:
        query_name (str): query_name as defined in tst.constants
        domain_size (int): domain size of the database instance
        itr_log_base (float): the base that gets a higher exponent each iteration
        itr_log_start (float): starting exponent of itr_log_base
        itr_log_end (float): final exponent of itr_log_base
        itr_num (int): The number of iterations in this data run
        bag_semantics (bool, optional). Indicates if bag semantics or set semantics is used. Defaults to False.
        max_bag_size (bool, optional). Indicates max bag size under bag semantics. Defaults to 10.
        output_filename (str, optional): Output storage csv
        overwrite_data (bool, optional): Indicates if output csv is overwritten. Defaults to False.
    """
    
    run_id = datetime.now().timestamp()
    query = queries[query_name]
    if query_name == 'hc-sj-ucq':
        query_del = [Query('R', ['x'], [('R', ['x', 'a','b']), ('R', ['x', 'b','c']), ('R', ['x', 'c', 'a'])]), 
                     Query('R', ['x'], [('R', ['x', 'e', 'f']), ('R', ['x', 'f', 'g'])])
                     ]
        query_min = [Query('R', ['x', 'a', 'b'], [('R', ['x', 'a', 'b'])])]
    else:
        query_del = [query]
        query_min = [Query(table_name, table_vars, [(table_name, table_vars)] ) for (table_name, table_vars) in query.query_body]

    database_instance = None
    tuple_weights = None

    tuples_added = 0
    
    current_itr = 0
    for itr_exponent in np.linspace(itr_log_start, itr_log_end, itr_num):
        current_itr += 1

        tuples_to_be_added = int(itr_log_base ** itr_exponent) - tuples_added

        if database_instance == None:
            database_instance, tuple_weights = addTuples(query.query_body,  tuples_to_be_added, domain_size, None, bag_semantics = bag_semantics)
        else:
            database_instance, tuple_weights = addTuples(query.query_body, tuples_to_be_added, domain_size, (database_instance, tuple_weights), bag_semantics = bag_semantics)

        tuples_added += tuples_to_be_added
        instance_id = datetime.now().timestamp()
        instance_data = (database_instance, tuple_weights)
        # database_instance = performSemijoinReduction(query, database_instance)

        pickle_filename = DATA_INSTANCE_PICKLE_FOLDER+str(instance_id)+'-r-'+str(run_id)+'.pkl'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(instance_data, f)

        output = {}

        output['instance timestamp'] = instance_id
        output['query'] = query_name
        output['domain size'] = domain_size
        output['run id'] = run_id
        output['itr exponent'] = itr_exponent
        output['itr start'] = itr_log_start
        output['itr end'] = itr_log_end
        output['itr num'] = itr_num
        output['tuples added'] = tuples_added
        output['bag semantics'] = bag_semantics 
        output['max bag size'] = max_bag_size if bag_semantics else 0
        output['processor'] = platform.processor()
        witnesses = {}
        for q in query_del:
            witnesses[q] = computeWitnesses(q, database_instance)
        output['number_of_witnesses'] = sum([len(witnesses[w]) for w in witnesses.keys()])
        projections = pd.DataFrame(columns=query.head_vars)
        for q in query_del:
            projections = pd.concat([projections, witnesses[q][query.head_vars]])
        projections = projections.drop_duplicates()

        expt_results = {}



        if k_in_percent:
            adpk = int((len(projections) * k)/ 100)
        else:
            adpk = k

        if adpk == 0:
            continue
        output['adpk'] = adpk
        
        expt_results["lp_results"] = gdp(database_instance, query_del = query_del, query_pres = None, kdel=[adpk], kpres=[], query_max = None, query_min = query_min, tuple_weights = tuple_weights, lp_type = "LP", kdel_is_total=True, treat_views_as_unions=True)
        if len(expt_results["lp_results"]) == 0:
            continue
        expt_results["ilp_results"] = gdp(database_instance, query_del = query_del, query_pres = None, kdel=[adpk], kpres=[], query_max = None, query_min = query_min, tuple_weights = tuple_weights, lp_type = "ILP", kdel_is_total=True, treat_views_as_unions=True)

        # expt_results["ilp_results_600"] = gdp(query, database_instance, tuple_weights = tuple_weights, lp_type = "ILP", time_limit = 600)
        # expt_results["ilp_results_60"] = gdp(query, database_instance, tuple_weights = tuple_weights, lp_type = "ILP", time_limit = 60)
        # expt_results["ilp_results_10"] = gdp(query, database_instance, tuple_weights = tuple_weights, lp_type = "ILP", time_limit = 10)

        for expt_name in expt_results:
            output.update({expt_name+': ' + str(key): val for key, val in expt_results[expt_name].items()})

        
        if not os.path.exists(output_filename) or overwrite_data:
            
            with open(output_filename, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=output.keys())
                writer.writeheader()
                writer.writerow(output)
            overwrite_data = False
        else:            
            
            with open(output_filename, 'a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=output.keys())
                writer.writerow(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a synthetic experiment')
    parser.add_argument('case', type=int, help="The case number to be run (Details specified in $EXPT_CASE_DETAILS_FILE)")
    args = parser.parse_args()
    runCase(args.case)