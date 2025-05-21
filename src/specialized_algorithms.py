from src.utils import computeWitnesses, get_existential_cc, isSingleton, isConnected, hasUniversalAttribute

import timeit
import pandas as pd

from src.query import Query
from src.resilience import resilience


def dp_view_side(query, database_instance, tuple_to_delete):
    """
    Given a query and a database instance, this function computes the min view side of deleting a prespecified output tuple

    Parameters:
    query (Query): The query for which the view side is to be computed
    database_instance (list of tuples): The database instance
    tuple_to_delete (dict): The tuple to be deleted

    Returns:
    results (dict): Contains various result parameters like the contingency set etc
    """

    results = {}
    # Step 1: Compute the witnesses
    # witness_compute_start_time = timeit.default_timer()
    # witnesses_df = computeWitnesses(query, database_instance)
    # witness_compute_end_time = timeit.default_timer()
    # results['witness_computation_time'] = witness_compute_end_time - witness_compute_start_time
    # Step 2: Compute the contingency set with the trivial algorithm
    solve_start_time = timeit.default_timer()
    contingency_set = []
    min_contingency_set_size = -1
    for (table_name, table_columns) in query.query_body:
        # Pick every tuple in database_instance that agrees with the tuple_to_delete on the table_columns
        table_df = pd.DataFrame(database_instance[table_name], columns=table_columns)

        # Find intersection of table_columns and head_vars
        head_variables_in_table = list(set(table_columns).intersection(set(query.head_vars)))

        # Select the tuples in the table that agree with the tuple_to_delete on head_variables_in_table
        test_contingency_set = table_df
        for v in head_variables_in_table:
            test_contingency_set = test_contingency_set[test_contingency_set[v] == tuple_to_delete[v]]

        if min_contingency_set_size == -1 or len(test_contingency_set) < min_contingency_set_size:
            min_contingency_set_size = len(test_contingency_set)
            contingency_set = test_contingency_set

    solve_end_time = timeit.default_timer()
    results['Solve Time'] = solve_end_time - solve_start_time
    results['contingency set'] = contingency_set
    results['dp-vs'] = min_contingency_set_size


    return results


def adp(query, database_instance, k):
    """
    TODO
    """
    return None

def smallest_witness_problem(query, database_instance):
    """
    Finds the smallest set of tuples that can be used to construct the same output view

    Parameters:
    query (Query): The query for which the view side is to be computed
    database_instance (list of tuples): The database instance

    Returns:
    results (dict): Contains various result parameters like the contingency set etc
    """
    results = {}
    tuples_to_keep = set()

    # Find the connected components of query
    connected_components = get_existential_cc(query)

    witnesses = computeWitnesses(query, database_instance)
    
    solve_start_time = timeit.default_timer()
    for i, cc in enumerate(connected_components):
        # Find head attributes of each connected component
        cc_vars = set()
        for (_, table_vars) in cc:
            cc_vars.update(table_vars)

        cc_head_vars = [v for v in cc_vars if v in query.head_vars]

        # Define a subquery with just the head variables in the connected component
        # compute the projection
        cc_projections = witnesses[cc_head_vars].drop_duplicates()

        for _, projection in cc_projections.iterrows():
            for (table_name, table_vars) in cc:
                # Keep a tuple from table_name that agrees with projection
                table = pd.DataFrame(database_instance[table_name], columns=table_vars)
                
                for v in projection.index:
                    table = table[table[v] == projection[v]] 

                tuple_to_keep = table_name +'_'+ '_'.join(str(x) for x in table.iloc[0].to_list())
                
                tuples_to_keep.add(tuple_to_keep)
    results['SWP'] = len(tuples_to_keep)
    solve_end_time = timeit.default_timer()
    results['Solve Time'] = solve_end_time - solve_start_time
    return results


def aggregated_deletion_propagation(query, database_instance, k):
    """
    Returns the aggregated deletion propagation for a given query and database instance with a given k

    Parameters:
    query (Query): The query computing the view on which deletion must be performed
    database_instance (list of tuples): The database instance
    k (int): The number of output tuples to delete

    Returns:
    results (dict): Contains various result parameters like the size of ADP etc
    """

    results = {}

    solve_start_time = timeit.default_timer()

    method_match = False

    print('Query:', query)

    if len(query.head_vars) == 0:
        method_match = True
        if k > 1:
            results['ADP'] = float('inf')
        else:
            results_resilience = resilience(query, database_instance, lp_type='LP')
            results['ADP'] = results_resilience['Resilience']
    
    else:
        is_singleton, singleton_table = isSingleton(query)
        if is_singleton:
            method_match = True
            results.update(aggregated_deletion_propagation_singleton(query, database_instance, k, singleton_table))
        
        else:
            is_universal, universal_attributes = hasUniversalAttribute(query)
            if is_universal:
                method_match = True
                results.update(aggregated_deletion_propagation_universal(query, database_instance, k, universal_attributes))

            num_cc, cc = isConnected(query)    
            if num_cc != 1:
                method_match = True
                results.update(aggregated_deletion_propagation_decomposition(query, database_instance, k, cc))


    if not method_match:
        results['error'] = 'ADP for this query '+str(query) +' cannot be found in PTIME via this method'

    solve_end_time = timeit.default_timer()
    results['Solve Time'] = solve_end_time - solve_start_time
    return results



def aggregated_deletion_propagation_singleton(query, database_instance, k, singleton_table):
    """
    Returns the aggregated deletion propagation for a given query and database instance with a given k

    Parameters:
    query (Query): The query computing the view on which deletion must be performed
    database_instance (list of tuples): The database instance
    k (int): The number of output tuples to delete
    singleton_table (str): A singleton table in the query

    Returns:
    results (dict): Contains various result parameters like the size of ADP etc
    """

    results = {}

    singleton_table_vars = [v[1] for v in query.query_body if v[0] == singleton_table][0]

    witnesses = computeWitnesses(query, database_instance)
    projections = witnesses[singleton_table_vars].drop_duplicates()
    # Case 1: Singleton variable has a subset of head variables

    singleton_table_list = list(database_instance[singleton_table])
    if set(singleton_table_vars).issubset(query.head_vars):
        cost = dict()
        for i, input_tuple in enumerate(singleton_table_list):
            # Find a cost of this input table which is equal to the number of tuples it corresponds to in the output view

            # Count the variables in the projection that agree with the input tuple
            cost[i] = 0
            for _, projection in projections.itertuples(name = None):
                projection = tuple([projection])
                if projection == input_tuple:
                    cost[i] += 1
            
        # Sort the cost dictionary in decreasing order
        costs_of_tuples = dict(sorted(cost.items(), key=lambda item: item[1], reverse = True))

        contingency_set = []
        output_deleted = 0
        for i in costs_of_tuples:
            if output_deleted >= k:
                break
            contingency_set.append(singleton_table + str(singleton_table_list[i]))
            output_deleted += cost[i]
        results['ADP'] = len(contingency_set)
        results['contingency set'] = contingency_set

        if output_deleted < k:
            results['ADP'] = float('inf')
            results['contingency set'] = []    

    return results


def aggregated_deletion_propagation_universal(query, database_instance, k, universal_attributes):
    """
    Returns the aggregated deletion propagation for a given query and database instance with a given k

    Parameters:
    query (Query): The query computing the view on which deletion must be performed
    database_instance (list of tuples): The database instance
    k (int): The number of output tuples to delete
    universal_attributes (str): Head variables that are present in all tables

    Returns:
    results (dict): Contains various result parameters like the size of ADP etc
    """

    # Split the database corresponding to different values of the universal attributes
    result = dict()
    witnesses = computeWitnesses(query, database_instance)

    database_instance_per_universal_projection = dict()
    for _, witness in witnesses.iterrows():
        universal_projection = "_".join([str(witness[u]) for u in universal_attributes])
        if universal_projection not in database_instance_per_universal_projection:
            database_instance_per_universal_projection[universal_projection] = {table_name:[] for table_name,_ in query.query_body}
        for (table_name, table_vars) in query.query_body:
            input_tuple = tuple(witness[var] for var in table_vars if var not in universal_attributes)
            # Also remove the universal attributes from the input tuple            
            database_instance_per_universal_projection[universal_projection][table_name].append(input_tuple)
            
            
    # Make everything a tuple
    for universal_projection in database_instance_per_universal_projection:
        for table_name in database_instance_per_universal_projection[universal_projection]:
            database_instance_per_universal_projection[universal_projection][table_name] = tuple(database_instance_per_universal_projection[universal_projection][table_name])

    # Find the new query without existential attributes
    new_query_body = [(table_name, [v for v in table_vars if v not in universal_attributes])for (table_name, table_vars) in query.query_body ]
    new_query = Query(query.name+'-u', list(set(query.head_vars) - set(universal_attributes)) , new_query_body)

    opt_adp = dict()
    opt_adp[0] = dict()
    # Compute the base case of ADP 
    projection_keys = list(database_instance_per_universal_projection.keys())
    for j in range(1, k+1):
        opt_adp[0][j] = aggregated_deletion_propagation(new_query, database_instance_per_universal_projection[projection_keys[0]], j)['ADP']

    for i in range(1, len(database_instance_per_universal_projection)):
        for j in range(1, k+1):
            if i not in opt_adp:
                opt_adp[i] = dict()
            opt_adp[i][j] = opt_adp[i-1][j]
            for m in range(1, j):
                c_im = aggregated_deletion_propagation(new_query, database_instance_per_universal_projection[projection_keys[i]], m)['ADP']
                if opt_adp[i][j] > opt_adp[i-1][j-m] + c_im:
                    opt_adp[i][j] = opt_adp[i-1][j-m] + c_im

    result['ADP'] = opt_adp[len(database_instance_per_universal_projection)-1][k]
    return result

def aggregated_deletion_propagation_decomposition(query, database_instance, k, connected_components):
    """
    Returns the aggregated deletion propagation for a given query and database instance with a given k

    Parameters:
    query (Query): The query computing the view on which deletion must be performed
    database_instance (list of tuples): The database instance
    k (int): The number of output tuples to delete
    connected_components (list): List of connected query components

    Returns:
    results (dict): Contains various result parameters like the size of ADP etc
    """

    # Split the database corresponding to different values of the universal attributes
    result = dict()

    # Get variables in query in first connected component
    cc0_vars = set()
    for (_, table_vars) in connected_components[0]:
        cc0_vars.update(table_vars)

    cc0_head_vars = [v for v in cc0_vars if v in query.head_vars]

    query_a = Query(query.name+'-cc0', cc0_head_vars, connected_components[0])

    opt_adp = dict()
    opt_adp[0] = dict()

    # Compute the base case of ADP
    for j in range(1, k+1):
        opt_adp[0][j] = aggregated_deletion_propagation(query_a, database_instance, j)['ADP']

    m1 = len(computeWitnesses(query_a, database_instance))

    for i in range(0, len(connected_components)):

        cc_i_vars = set()
        for (_, table_vars) in connected_components[i]:
            cc_i_vars.update(table_vars)
        cc_i_head_vars = [v for v in cc_i_vars if v in query.head_vars]
        query_i = Query(query.name+'-cc'+str(i), cc_i_head_vars, connected_components[i])
        
        m2 = len(computeWitnesses(query_i, database_instance))
        for j in range(1, k+1):
            if i not in opt_adp:
                opt_adp[i] = dict()
            opt_adp[i][j] = float('inf')
            for k1 in range(0, j+1):
                for k2 in range(0, j+1):
                    if k1*m1 + k2*m2 - k1*k2 >= j:
                        
                        cik2 = aggregated_deletion_propagation(query_i, database_instance, k2)['ADP']
                        if opt_adp[i][j] > opt_adp[i-1][k1] + cik2:
                            opt_adp[i][j] = opt_adp[i-1][k1] + cik2
        query_a = Query(query.name+'-cc-0-'+str(i), list(set(cc_i_head_vars)+set(cc0_head_vars)), connected_components[0]+connected_components[i])
        m1 += m2
    
    result['ADP'] = opt_adp[len(connected_components)-1][k]

    return result

