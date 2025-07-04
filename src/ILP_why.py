import gurobipy as gp
from gurobipy import GRB
from sympy import symbols, Equivalent, to_cnf, Or, And, Not
from benchmark_models import get_sympy_SMK

# This file reproduces the algorithm from https://github.com/amjadKhalifah/HP2SAT1.0/tree/hp-optimization-library

#### Sympy setup
# Extract clauses from the CNF formula
def extract_clauses(expr):
    """Extract clauses from a CNF expression."""
    if isinstance(expr, Or):  # A single disjunction (clause)
        return [expr]
    elif isinstance(expr, And):  # Multiple clauses
        return list(expr.args)
    else:  # A single literal
        return [expr]

# Convert the symbolic clauses to integers for the SAT solver
def clause_to_str(clause):
    """Convert a symbolic clause to integers."""
    if isinstance(clause, Or):  # Disjunction
        return [lit_to_str(arg) for arg in clause.args]
    else:  # Single literal
        return [lit_to_str(clause)]

def lit_to_str(literal):
    """Convert a symbolic literal to an integer."""
    if literal.is_Symbol:  # Positive literal
        return literal.name  # Assume variable names are x1, x2, etc.
    elif isinstance(literal, Not):  # Negated literal
        return "~" + literal.args[0].name
    else:
        raise ValueError(f"Unexpected literal format: {literal}")

def get_identifiers(clauses_str):
    vars = list({var if not var.startswith("~") else var[1:] for clause in clauses_str for var in clause})
    return dict(zip(vars, range(1, len(vars)+1)))

def clauses_str_to_int(clauses_str, var2int):
    return [[var2int[var] if not var.startswith("~") else -var2int[var[1:]] for var in clause] for clause in clauses_str]

def show_model(model, var2int):
    int2var = [None] + [key for key, value in sorted(var2int.items(), key=lambda x: x[1])]
    for value in model:
        var = int2var[abs(value)]
        if var.upper() == var: 
            print(f"{var}={1 if value == abs(value) else 0}", end=", ")
    print("\b\b")

def get_int_cnf_from_formula(formula):
    if len(formula.atoms()) > 7: print(f"Careful: {len(formula.atoms())} variables (recommended <7)")
    cnf_formula = to_cnf(formula, simplify=True, force=True)
    clauses = extract_clauses(cnf_formula)
    str_clauses = [clause_to_str(cl) for cl in clauses]
    var2int = get_identifiers(str_clauses)
    int_clauses = clauses_str_to_int(str_clauses, var2int)
    return int_clauses, var2int

def get_int_cnf_from_list(formulas):
    str_clauses = []
    for formula in formulas:
        if len(formula.atoms()) > 7: print(f"Careful: {len(formula.atoms())} variables (recommended <7)")
        cnf_formula = to_cnf(formula, simplify=True, force=True)
        clauses = extract_clauses(cnf_formula)
        str_clauses += [clause_to_str(cl) for cl in clauses]
    var2int = get_identifiers(str_clauses)
    int_clauses = clauses_str_to_int(str_clauses, var2int)
    return int_clauses, var2int

def get_formula_infering(symbs, target, x, u, equations, verbose=0):
    str2var = {var.name: var for var in symbs}
    x = {lit(var): sign_lit(var, str2var) for var in x.split()}
    u = {lit(var): sign_lit(var, str2var) for var in u.split()}


    if verbose: print(~target)
    F = [~target]
    for var_str, var in str2var.items():
        # if var.name == target.name: continue
        if var_str in u:
            if verbose: print(u[var_str])
            F += [u[var_str]]
        elif var.name == target.name:
            F += [Equivalent(var,equations[var])]
        elif var_str in x:
            C1_i = symbols("C1_"+var_str) 
            C2_i = symbols("C2_"+var_str) 
            clause_fnt = (Equivalent(var, equations[var]) & C1_i) | (~Equivalent(var, equations[var]) & ~C1_i)
            clause_ori = (x[var_str] & C2_i) | (~x[var_str] & ~C2_i)
            if verbose: print(clause_fnt)
            if verbose: print(clause_ori)
            F += [clause_fnt] + [clause_ori]
            
    if verbose: print()
    return F

def lit(var):
    return var if not var.startswith('~') else var[1:]

def sign_lit(var, str2var):
    return ~str2var[lit(var)] if var.startswith('~') else str2var[lit(var)]

#### Gerubi optimization
def get_gerubi_options(prefix=""):
    with open(prefix+"gurobi.lic") as file:
        lines = file.read().split("\n")
    options = dict(map(lambda line: line.split("="),lines[-4:-1]))
    options["LICENSEID"] = int(options["LICENSEID"])
    return options

    
def get_causes_from_model(model, sp_vars, target):
    C, W = [], []
    for var in sp_vars:
        var_name = var.name
        if var.name == target.name: continue
        c1 = model.getVarByName("C1_" + var_name).x
        c2 = model.getVarByName("C2_" + var_name).x
        if (c1,c2) == (0,0): 
            C.append(var)
        elif (c1,c2) == (0,1): 
            W.append(var)
    return C, W

def add_cnf_to_model(model, cnf):
    if cnf.is_Atom:
        var = model.getVarByName(cnf.name)
        model.addConstr(var == 1, str(cnf))
        return 
    if cnf.is_Not:
        sp_var = cnf.args[0]
        var = model.getVarByName(sp_var.name)
        model.addConstr(var == 0, str(cnf))
        return 
    for clause in cnf.args:
        add_clause_to_model(model, clause)

def add_clause_to_model(model, clause):
    gb_clause = []
    for literal in clause.args:
        if literal.is_Atom:
            gb_clause.append(model.getVarByName(literal.name))
        else:
            sp_var = literal.args[0]
            gb_clause.append(1 - model.getVarByName(sp_var.name))
    model.addConstr(gp.quicksum(gb_clause) >= 1.0)

def add_vars_to_model(model, sp_vars, u, target):
    model.ModelSense = -1
    c1SumExp = 0
    c3SumExp = 0
    for var_name in u.split(" "):
        model.addVar(vtype=GRB.BINARY, name=var_name.replace("~", ""))
    for v in sp_vars:
        model.addVar(vtype=GRB.BINARY, name=v.name)
        if v.name != target.name:
            c1 = model.addVar(vtype=GRB.BINARY, name="C1_" + v.name)
            c2 = model.addVar(vtype=GRB.BINARY, name="C2_" + v.name)
            c3 = model.addVar(vtype=GRB.BINARY, name="C3_" + v.name)
            # c3 is an additional var based on c1 and c2 used for simplicity
            # c3 = ¬c1 & ¬c
            model.addConstr(- c1 - c2 - 2 * c3 <= -1.0, "c3_" + v.name + "_1")
            model.addConstr(- c1 - c2 - 2 * c3 >= -2.0, "c3_" + v.name + "_2")

            # prepare the c1 and c3 sums
            c1SumExp = c1SumExp + c1
            c3SumExp = c3SumExp + c3

    model.update()
    # C -> C1,C2 = 0,0 -> C3 = 1
    # W -> C1,C2 = 0,1
    # C3 = ¬C1 & ¬C2
    # Normal -> C1 = 1
    model.setObjectiveN(
        expr=c1SumExp, 
        index=0, 
        priority=2, 
        weight=1, 
        abstol=0, 
        reltol=0, 
        name="Normal variables")
    
    model.setObjectiveN(
        expr = -c3SumExp, 
        index=1, 
        priority=1, 
        weight=1, 
        abstol=0, 
        reltol=0, 
        name="Cause variables")

def ilp_why(endo_variables, u, target, G_cnf, prefix, verbose):
    with gp.Env(empty=True) as env:
        params = get_gerubi_options(prefix)
        for param, value in params.items():
            env.setParam(param, value)
        env.setParam('OutputFlag', 0)
        env.setParam('LogToConsole', 0)
        env.start()
        with gp.Model(env=env) as model:
            add_vars_to_model(model, endo_variables, u, target)
            for cnf in G_cnf:
                add_cnf_to_model(model, cnf)
            var = model.getVarByName("C2_BH")
            model.update()
            model.optimize()
        
            # Print the results
            if model.status == GRB.OPTIMAL:
                if verbose:
                    for i in range(2):
                        obj = model.getObjective(i)
                        print(f"Obj {i}:", obj.getValue())
                    show_all_vars(model, endo_variables, target)
                C, W = get_causes_from_model(model, endo_variables, target)
                return C, W
            else:
                raise Exception("No solution")

##### Main
def run_ilp_SMK(n_attacker, instance, V_exo, prefix="../"):
    sp_vars, equations, target = get_sympy_SMK(n_attacker)
    x = " ".join(["~"*(1-value) + var.name
                  for var, value in zip(sp_vars, instance)])
    u = " ".join(["~"*(1-value) + var.name.lower() 
                  for var, value in zip(sp_vars, V_exo)])
    G = get_formula_infering(sp_vars, target, x, u, equations)
    G_cnf = [to_cnf(g) for g in G]
    C, W = ilp_why(equations.keys(), u, target, G_cnf, prefix, False)
    return C, W