'''
Find resilience of a set of witnesses under a query
'''

import pulp
import timeit

from src.utils import computeWitnesses, addLPVariable

def resilience(query, database_instance, tuple_weights = {}, time_limit = None, lp_type = 'ILP', verbosity = 0, exogenous_tables=[]):
    '''
    Calculates the resilience of given database instance using an ILP

    Args:
        query (list): A boolean conjunctive query described by the variables in each table
        database_instance (list): A list of tuples present in each table
        tuple_weights (list, optional): Weight of tuples under bag semantics. Defaults to an empty dict.
        time_limit (int, optional): The timeout period of the optimization. Defaults to None.
        lp_type (str, optional): Whether to perform an ILP or LP optimization. Defaults to 'ILP'.
        verbosity (int, optional): Level of verbosity of the output. Defaults to 1.
        exogenous_tables(list, optional): Tables which may not be removed in resilience computation. Defaults to empty

    Returns:
        result: A dictionary with result and problem parameters -> resilience, solve time etc
    '''

    result = {}


    witness_computation_start_time = timeit.default_timer()
    witnesses = computeWitnesses(query, database_instance)
    result['witness computation time'] = timeit.default_timer() - witness_computation_start_time
    result['number of witnesses'] = len(witnesses)

    query = query.query_body
    if len(witnesses) == 0:
        return {}

    
    witnesses_map = []
    for i in range(len(witnesses.index)):
        w_map = {}
        for col in witnesses:
            w_map[col] = '_'+str(witnesses[col].loc[i])
        witnesses_map.append(w_map)

    
    tuple_variables = {}
    for w in witnesses_map:
        for (table_name, table_columns) in query:
            variableKey = table_name +''.join([w[variable] for variable in table_columns])
            addLPVariable(variableKey, tuple_variables, lp_type = lp_type)

    
    
    prob = pulp.LpProblem('Resilience', pulp.LpMinimize)

    
    prob += pulp.lpSum((tuple_weights[t] if t in tuple_weights else 1)*tuple_variables[t] for t in tuple_variables)

    
    for w in witnesses_map:
        tuples = []
        for (table_name, table_columns) in query:
            if table_name not in exogenous_tables:
                t = tuple_variables[table_name + ''.join([w[variable] for variable in table_columns])]
                tuples.append(t)

        tuples = set(tuples) 
        prob += pulp.lpSum(tuples) >= 1

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
        
        if lp_type == 'LP':
            k = len(query)
            approx_obj = 0
            for v in prob.variables():
                if v.varValue >= (1/k):
                    if v.name in tuple_weights:
                        approx_obj += 1 * tuple_weights[v.name]
                    else:
                        approx_obj += 1
        
            result['resilience lp approximation'] = approx_obj
        
        result['Resilience'] = pulp.value(prob.objective) 
        result['error'] = None

    except Exception as e:
        print('Error!')
        print(e)
        result['Solve Time'] = -1
        result['Solver Solution Time'] = -1
        if lp_type == 'LP':
            result['resilience lp approximation'] = -1
        result['Resilience'] = -1 
        result['error'] = str(e)

    return result