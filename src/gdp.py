import timeit
import pulp
import pandas as pd

from src.constants import queries
from src.utils import computeWitnesses, representWitnessAsList, representProjectionAsList, addLPVariable

def gdp_old_formulation(database_instance, query_del = None, kdel = None, query_pres = None, kpres =None, query_min = None, query_max = None, tuple_weights = {}, time_limit = None, lp_type = 'ILP', verbosity = 0):
    '''
    Calculates the resilience of given database instance using an ILP

    This ILP does not recover LP = ILP for SWP!!

    Args:
        database_instance (list): A list of tuples present in each table
        query_del (list of Query, optional) : Queries constructing the views from which tuples should be deleted
        kdel (list, optional): Number of tuples to be deleted. Defaults to None.
        query_pres (list of Query, optional) : Queries constructing the views from which tuples should be preserved
        kpres (list, optional): Number of tuples to be preserved. Defaults to None.
        query_min (list of Query, optional) : Queries constructing the views from which deletion of tuples should be minimized
        query_max (list of Query, optional) : Queries constructing the views from which deletion of tuples should be maximized
        tuple_weights (list, optional): Weight of tuples under bag semantics. Defaults to an empty dict.
        time_limit (int, optional): The timeout period of the optimization. Defaults to None.
        lp_type (str, optional): Whether to perform an ILP or LP optimization. Defaults to 'ILP'.
        verbosity (int, optional): Level of verbosity of the output. Defaults to 1.

    Returns:
        result: A dictionary with result and problem parameters -> resilience, solve time etc
    '''

    result = {}
    # Step 1: Compute the witnesses
    queries = {'del': query_del, 'pres': query_pres, 'min': query_min, 'max': query_max}
    witnesses_list = {'del':[], 'pres':[], 'min':[], 'max':[]}
    witnesses_dfs = {'del':[], 'pres':[], 'min':[], 'max':[]}
    projections_list = {'del':[], 'pres':[], 'min':[], 'max':[]}
    projection_dfs = {'del':[], 'pres':[], 'min':[], 'max':[]}

    witness_computation_start_time = timeit.default_timer()
    for query_type in queries:
        if queries[query_type] is not None:
            for query in queries[query_type]:
                witnesses_df = computeWitnesses(query, database_instance)
                witnesses_list[query_type].append(representWitnessAsList(query, witnesses_df))
                witnesses_dfs[query_type].append(witnesses_df)

                projection_df = witnesses_df[query.head_vars]
                projection_dfs[query_type].append(projection_df)
                projections_list[query_type].append(representProjectionAsList(query, witnesses_df))

                if len(witnesses_df) == 0:
                    return result

    
    witness_computation_end_time = timeit.default_timer()
    witness_computation_time = witness_computation_end_time - witness_computation_start_time
    result['witness_computation_time'] = witness_computation_time

    list_of_tuples = set()
    
    
    # Step 2: Build the ILP
    prob = pulp.LpProblem('GDP', pulp.LpMinimize)

    # Step 2a: Add ILP variables - for each tuple, each witness and each projection
    ilp_variables = dict()

    for query_type in queries:
        if queries[query_type] is not None:
            for witneses in witnesses_list[query_type]:
                for witness in witneses:
                    # Add witness variables
                    addLPVariable('w-'+'-'.join(witness), ilp_variables, lp_type)
                    for tuple_i in witness:
                        # Add tuple variables
                        addLPVariable(tuple_i, ilp_variables, lp_type)
                        list_of_tuples.add(tuple_i)

    result['number_of_tuples'] = len(list_of_tuples)

    # Add projection variables
    for query_type in queries:
        if queries[query_type] is not None:
            for i, query in enumerate(queries[query_type]):
                for projection_i in projections_list[query_type][i].keys():
                    addLPVariable(projection_i, ilp_variables, lp_type)

    prob += pulp.lpSum([(tuple_weights[projection_i] if projection_i in tuple_weights else 1) * ilp_variables[projection_i] for i in range(len(projections_list['min'])) for projection_i in projections_list['min'][i].keys()]) 
    -  pulp.lpSum([ilp_variables[projection_i] for i in range(len(projections_list['max'])) for projection_i in projections_list['max'][i].keys()])


    # Step 2b: Add hard user constraints
    # Deletion constraints
    if queries['del'] is not None:
        for i, query in enumerate(queries['del']):
            # At least kdeli projections must be deleted
            prob += pulp.lpSum([ilp_variables[projection_i] for projection_i in projections_list['del'][i]]) >= kdel[i]

    # Preservation constraints
    if queries['pres'] is not None:
        for i, query in enumerate(queries['pres']):
            # At least kpresi projections must be preserved - less than kpresi projections must be deleted
            prob += pulp.lpSum([ilp_variables[projection_i] for projection_i in projections_list['pres'][i]]) <= len(projections_list['pres'][i]) - kpres[i]
    
    
    # Step 2b: Add propagation constraints
    # PC 1: A witness is deleted when any of its tuples are deleted - applies to only view_pres and view_max
    for query_type in ['pres', 'max']:
        if queries[query_type] is not None:
            for query in queries[query_type]:
                for witness in witnesses_list[query_type][i]:
                    for tuple_i in witness:
                        prob += ilp_variables[tuple_i] <= ilp_variables['w-'+'-'.join(witness)]

    # PC 2: A witness cannot be deleted if any of its tuples are not deleted - applies to only view_del and view_min
    for query_type in ['del', 'min']:
        if queries[query_type] is not None:
            for query in queries[query_type]:
                for witness in witnesses_list[query_type][i]:
                    prob += ilp_variables['w-'+'-'.join(witness)] <= pulp.lpSum([ilp_variables[tuple_i] for tuple_i in witness])

    # PC 3: A projection is deleted if all corresponding witnesses are deleted 
    for query_type in ['pres', 'max']:
        if queries[query_type] is not None:
            for i, query in enumerate(queries[query_type]):
                for projection_j in projections_list[query_type][i].keys():
                    prob += 1 + pulp.lpSum([ilp_variables[witness_k] for witness_k in projections_list[query_type][i][projection_j]]) - len(projections_list[query_type][i][projection_j]) <= ilp_variables[projection_j] 

    # PC 4: If a projection is deleted, all corresponding witnesses are deleted 
    for query_type in ['del', 'min']:
        if queries[query_type] is not None:
            for i, query in enumerate(queries[query_type]):
                for projection_j in projections_list[query_type][i].keys():
                    for witness_k in projections_list[query_type][i][projection_j]:
                        prob += ilp_variables[projection_j] <= ilp_variables[witness_k]

    # Step 3: Solve the ILP
    if verbosity >= 1:
        print(prob)

    try:
        
        ilp_solve_begin_time = timeit.default_timer()
        if time_limit is None:
            prob.solve(pulp.GUROBI_CMD())
        else:
            prob.solve(pulp.GUROBI_CMD(options=[('TimeLimit',time_limit)]))  
        result['Solve Time'] = timeit.default_timer() - ilp_solve_begin_time
        result['Solver Solution Time'] = prob.solutionTime

        if verbosity >= 1:
            print('Status:', pulp.LpStatus[prob.status])

        if verbosity >= 1:
            for v in prob.variables():
                if v.varValue == 1:
                    print(v.name, '=', v.varValue)

        result['GDP'] = pulp.value(prob.objective) 
        result['error'] = None


    except Exception as e:
        print('Error!')
        print(e)
        result['Solve Time'] = -1
        result['Solver Solution Time'] = -1
        if lp_type == 'LP':
            result['GDP lp approximation'] = -1
        result['GDP'] = -1 
        result['error'] = str(e)
    
        
    # Step 4: Return the result

    return result

def gdp(database_instance, query_del = None, kdel = None, query_pres = None, kpres =None, query_min = None, query_max = None, tuple_weights = {}, time_limit = None, lp_type = 'ILP', verbosity = 0, old_ilp_formulation = False, kdel_is_total = False, treat_views_as_unions = False, allow_empty = False):
    '''
    Calculates the resilience of given database instance using an ILP

    Args:
        database_instance (list): A list of tuples present in each table
        query_del (list of Query, optional) : Queries constructing the views from which tuples should be deleted
        kdel (list, optional): Number of tuples to be deleted. Defaults to None.
        query_pres (list of Query, optional) : Queries constructing the views from which tuples should be preserved
        kpres (list, optional): Number of tuples to be preserved. Defaults to None.
        query_min (list of Query, optional) : Queries constructing the views from which deletion of tuples should be minimized
        query_max (list of Query, optional) : Queries constructing the views from which deletion of tuples should be maximized
        tuple_weights (list, optional): Weight of tuples under bag semantics. Defaults to an empty dict.
        time_limit (int, optional): The timeout period of the optimization. Defaults to None.
        lp_type (str, optional): Whether to perform an ILP or LP optimization. Defaults to 'ILP'.
        verbosity (int, optional): Level of verbosity of the output. Defaults to 1.

    Returns:
        result: A dictionary with result and problem parameters -> resilience, solve time etc
    '''

    result = {}

    # Step 1: Compute the witnesses
    queries = {'del': query_del, 'pres': query_pres, 'min': query_min, 'max': query_max}
    witnesses_list = {'del':[], 'pres':[], 'min':[], 'max':[]} # Witnesses as a list of tuples e.g. {'w-A_1-R_1_2-S_1_2': ['x_1', 'x_2'], 'w-A_3-R_3_4-S_3_4': ['x_3', 'x_4']}
    witnesses_dfs = {'del':[], 'pres':[], 'min':[], 'max':[]} #
    projections_list = {'del':[], 'pres':[], 'min':[], 'max':[]} # Projections as a list of witnesses e.g. {'p-x_1': ['w-A_1-R_1_2-S_1_2'], 'p-x_3': ['w-A_3-R_3_4-S_3_4']}
    projection_dfs = {'del':[], 'pres':[], 'min':[], 'max':[]}

    witness_computation_start_time = timeit.default_timer()
    for query_type in queries:
        if queries[query_type] is not None:
            for query in queries[query_type]:
                witnesses_df = computeWitnesses(query, database_instance)
                witnesses_list[query_type].append(representWitnessAsList(query, witnesses_df))
                witnesses_dfs[query_type].append(witnesses_df)

                projection_df = witnesses_df[query.head_vars]
                projections_list[query_type].append(representProjectionAsList(query, witnesses_df))
                projection_dfs[query_type].append(projection_df)

                print(len(witnesses_df), query_type, query)

                if len(witnesses_df) == 0 and not allow_empty:
                    return result
                
    
    witness_computation_end_time = timeit.default_timer()
    witness_computation_time = witness_computation_end_time - witness_computation_start_time
    result['witness_computation_time'] = witness_computation_time
    result['number_of_witnesses'] = sum([len(witnesses_i) for query_type in queries if queries[query_type] is not None for witnesses_i in witnesses_list[query_type] ])
    
    # Step 2: Build the ILP
    prob = pulp.LpProblem('GDP', pulp.LpMinimize)

    # Step 2a: Add ILP variables - for each tuple, each witness and each projection
    ilp_variables = dict()

    list_of_tuples = set()

    for query_type in queries:
        if queries[query_type] is not None:
            for view_id, witneses in enumerate(witnesses_list[query_type]):
                view_key = query_type + '-' + str(view_id)
                if treat_views_as_unions:
                    view_key = query_type
                for witness in witneses:
                    # Add witness variables
                    addLPVariable(view_key+'-w-'+'-'.join(witness), ilp_variables, lp_type)
                    for tuple_i in witness:
                        # Add tuple variables
                        addLPVariable(tuple_i, ilp_variables, lp_type)
                        list_of_tuples.add(tuple_i)

    
    result['number_of_tuples'] = len(list_of_tuples)

    # Add projection variables
    for query_type in queries:
        if queries[query_type] is not None:
            for view_id, query in enumerate(queries[query_type]):
                view_key = query_type + '-' + str(view_id)
                if treat_views_as_unions:
                    view_key = query_type
                for projection_i in projections_list[query_type][view_id].keys():
                    addLPVariable(view_key+'-'+projection_i, ilp_variables, lp_type)
    
    if treat_views_as_unions:
        prob += pulp.lpSum([(tuple_weights[projection_i] if projection_i in tuple_weights else 1) * ilp_variables['min-'+projection_i] for i in range(len(projections_list['min'])) for projection_i in projections_list['min'][i].keys()]) -  pulp.lpSum([(tuple_weights[projection_i] if projection_i in tuple_weights else 1) * ilp_variables['max-'+projection_i] for i in range(len(projections_list['max'])) for projection_i in projections_list['max'][i].keys()])

    else:
        prob += pulp.lpSum([(tuple_weights[projection_i] if projection_i in tuple_weights else 1) * ilp_variables['min-'+str(i)+'-'+projection_i] for i in range(len(projections_list['min'])) for projection_i in projections_list['min'][i].keys()]) -  pulp.lpSum([(tuple_weights[projection_i] if projection_i in tuple_weights else 1) * ilp_variables['max-'+str(i)+'-'+projection_i] for i in range(len(projections_list['max'])) for projection_i in projections_list['max'][i].keys()])


    # Step 2b: Add constraints
    
    # For view del
    if queries['del'] is not None:
        query_type = 'del'
        if kdel_is_total:
            if treat_views_as_unions:
                lp_vars = dict()
                for i in range(len(projections_list['del'])):
                    for projection_i in projections_list['del'][i]:
                        lp_vars[ilp_variables['del-'+projection_i]] = tuple_weights[projection_i] if projection_i in tuple_weights else 1

                prob += pulp.lpSum([lp_vars[l]*l for l in lp_vars]) >= kdel
            else:
                prob += pulp.lpSum(list(set([ilp_variables['del-'+str(i)+'-'+projection_i] for i in range(len(projections_list['del'])) for projection_i in projections_list['del'][i]]))) >= kdel

        for i, query in enumerate(queries['del']):
            view_key = 'del-'+str(i)+'-'
            if treat_views_as_unions:
                view_key = "del-"

            # Hard user constraint
            # At least kdeli projections must be deleted
            if not kdel_is_total:
                prob += pulp.lpSum([ilp_variables[view_key+projection_i] for projection_i in projections_list['del'][i]]) >= kdel[i]

            # Propagate view to witness
            for projection_j in projections_list[query_type][i].keys():
                for witness_k in projections_list[query_type][i][projection_j]:
                    prob += ilp_variables[view_key+projection_j] <= ilp_variables[view_key+witness_k]

            # Propagate witness to tuple
            for witness in witnesses_list[query_type][i]:
                    prob += ilp_variables[view_key+'w-'+'-'.join(witness)] <= pulp.lpSum([ilp_variables[tuple_i] for tuple_i in witness])

    # For view pres
    if queries['pres'] is not None:
        query_type = 'pres'
        for i, query in enumerate(queries['pres']):
            view_key = 'pres-'+str(i)+'-'
            if treat_views_as_unions:
                view_key = "pres-"

            # Hard user constraint
            # At least kpresi projections must be preserved - less than kpresi projections must be deleted
            prob += pulp.lpSum([(tuple_weights[projection_i] if projection_i in tuple_weights else 1) * ilp_variables[view_key+projection_i] for projection_i in projections_list['pres'][i]]) <= len(projections_list['pres'][i]) - kpres[i]

            # Propagate view to witness
            for projection_j in projections_list[query_type][i].keys():
                prob += 1 + pulp.lpSum([ilp_variables[view_key+witness_k] for witness_k in projections_list[query_type][i][projection_j]]) - len(projections_list[query_type][i][projection_j]) <= ilp_variables[view_key+projection_j] 


            # Propagate witness to tuple
            if old_ilp_formulation: 
                for witness in witnesses_list[query_type][i]:
                    for tuple_i in witness:
                        prob += ilp_variables[tuple_i] <= ilp_variables[view_key+'w-'+'-'.join(witness)]

            else:
                witnesses_list_per_tuple = {}
                # Find all the witnesses associated with each tuple
                for witness in witnesses_list[query_type][i]:
                    for tuple_i in witness:
                        if tuple_i not in witnesses_list_per_tuple:
                            witnesses_list_per_tuple[tuple_i] = [view_key+'w-'+'-'.join(witness)]
                        else:
                            witnesses_list_per_tuple[tuple_i].append(view_key+'w-'+'-'.join(witness))

                # Add constraint for each tuple
                for tuple_i in witnesses_list_per_tuple:
                    prob += ilp_variables[tuple_i] <= 1 + pulp.lpSum([ilp_variables[witness] for witness in  witnesses_list_per_tuple[tuple_i]]) - len(witnesses_list_per_tuple[tuple_i])


    # For view min
    if queries['max'] is not None:
        query_type = 'max'
        for i, query in enumerate(queries['max']):
            view_key = 'max-'+str(i)+'-'
            if treat_views_as_unions:
                view_key = "max-"

            # Propagate view to witness
            for projection_j in projections_list[query_type][i].keys():
                for witness_k in projections_list[query_type][i][projection_j]:
                    prob += ilp_variables[view_key+projection_j] <= ilp_variables[view_key+witness_k]

            # Propagate witness to tuple
            for witness in witnesses_list[query_type][i]:
                    prob += ilp_variables[view_key+'w-'+'-'.join(witness)] <= pulp.lpSum([ilp_variables[tuple_i] for tuple_i in witness])



    # For view max
    if queries['min'] is not None:
        query_type = 'min'
        for i, query in enumerate(queries['min']):
            view_key = 'min-'+str(i)+'-'
            if treat_views_as_unions:
                view_key = "min-"

            # Propagate view to witness
            for projection_j in projections_list[query_type][i].keys():
                prob += 1 + pulp.lpSum([ilp_variables[view_key+witness_k] for witness_k in projections_list[query_type][i][projection_j]]) - len(projections_list[query_type][i][projection_j]) <= ilp_variables[view_key+projection_j] 

            # Propagate witness to tuple
            for witness in witnesses_list[query_type][i]:
                for tuple_i in witness:
                    prob += ilp_variables[tuple_i] <= ilp_variables[view_key+'w-'+'-'.join(witness)]

    

    # Step 3: Solve the ILP
    if verbosity >= 1:
        print(prob)

    try:
        
        ilp_solve_begin_time = timeit.default_timer()
        if time_limit is None:
            prob.solve(pulp.GUROBI_CMD())
        else:
            prob.solve(pulp.GUROBI_CMD(options=[('TimeLimit',time_limit)]))  
        result['Solve Time'] = timeit.default_timer() - ilp_solve_begin_time
        result['Solver Solution Time'] = prob.solutionTime

        if verbosity >= 1:
            print('Status:', pulp.LpStatus[prob.status])

        if verbosity >= 1:
            for v in prob.variables():
                if v.varValue != 0:
                    print(v.name, '=', v.varValue)

        result['GDP'] = pulp.value(prob.objective) 
        result['error'] = None


    except Exception as e:
        print('Error!')
        print(e)
        result['Solve Time'] = -1
        result['Solver Solution Time'] = -1
        if lp_type == 'LP':
            result['GDP lp approximation'] = -1
        result['GDP'] = -1 
        result['error'] = str(e)
    
        
    # Step 4: Return the result

    return result