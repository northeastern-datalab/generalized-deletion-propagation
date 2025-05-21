# Define a python class that represents a conjunctive query.

class Query:

    def __init__(self, name, head_vars, query_body, constants= None):
        '''
        name: str
        head_vars: list of str
        query_body: list of tuple where each tuple is a relation name and a list of variables
        '''
        self.name = name
        self.head_vars = head_vars
        self.query_body = query_body
        self.constants = constants

    def __str__(self):
        return self.name + "(" + ",".join(self.head_vars) + ") :- " + ", ".join([rel + "(" + ",".join(vars) + ")" for rel, vars in self.query_body])
    
    def __repr__(self):
        return self.__str__()