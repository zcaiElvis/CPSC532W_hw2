# Standard imports
import torch as tc
import json

# Project imports
from primitives import primitives # NOTE: Import and use this!



class abstract_syntax_tree:
    def __init__(self, ast_json):
        self.ast_json = ast_json
        self.functions = ast_json[:-1]
        self.program = ast_json[-1]


def add_functions(j):
    rho = {}
    for fun in j:
        fname = j[0][1]
        fvars = j[0][2]
        fexpr = j[0][3]
        rho[fname] = [fexpr, fvars]
    return rho


def evaluate(j, l={}, rho={}, sigma=0):

    if isinstance(j, bool):
        return tc.tensor(int(j)).float()

    elif isinstance(j, int) or isinstance(j, float):
        return tc.tensor(j).float()

    elif j[0] == "observe":
        d = evaluate(j[1], l = l, rho = rho)
        v = evaluate(j[2], l=l, rho = rho)
        logp = d.log_prob(v)
        sigma+= tc.tensor(logp)
        return v

    elif j[0] == "sample" or j[0] == "sample*":
        return evaluate(j[1], l = l, rho=rho).sample()

    elif j[0] == "if":
        true_or_not = evaluate(j[1], l = l , rho= rho)
        if(true_or_not):
            return evaluate(j[2], l = l, rho=rho)
        else:
            return evaluate(j[3], l = l, rho=rho)


    elif j[0] == "let":
        c = evaluate(j[1][1], l, rho)
        l[j[1][0]] = c
        return evaluate(j[2], l, rho)


    elif isinstance(j, str):

        if j in rho:
            return evaluate(rho[j], l=l, rho=rho)

        # if isinstance(rho[j], list):
        #     return evaluate(l[j], l = l, rho=rho)
        else:
            return l[j]

    else:
        opt = rho[j[0]]

        values = []
        for i in range(1, len(j)):
            values.append(evaluate(j[i], l, rho))

        if isinstance(opt, list):
            # if opt is a list like (* var var)
            fvars = rho[j[0]][1]
            fexpr = rho[j[0]][0]
            localenv = dict(zip(fvars, values))
            return evaluate(fexpr, l = localenv, rho = rho)

        return opt(*values)



def evaluate_program(ast, verbose=False):

    env = add_functions(ast.functions)
    env = {**env, **primitives}
    result = evaluate(ast.program, rho = env)

    return result, None, None # NOTE: This should (artifically) pass deterministic test 1
