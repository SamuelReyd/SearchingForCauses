class SCM:
    def __init__(self, V:list, U:list, D:list[list], F:callable, u:list, 
                 psi:callable, dag:list[list]=None, sim:callable=None):
        """
        SCM object that includes everything needed to execute the algorithms
            V: list of endogenous variables labels (the target must be the last variable)
            U: list of exogenous variables labels
            D: list of discrete domains for the variables in V (in the same order)
            F: function that takes an intervention and a context as input and outputs v
            u: context (values of U in the right order)
            psi: heuristic to evaluate the quality of an intervention, takes v as input and outputs a float
            sim (optional): when F and psi are None, sim can be provided instead. 
                Input is a set of interventions.
                Output is a list with a value for each intervention. 
                Each value of the output is a tuple (state_info, phi, psi).
                This argument is a more general approach when F is harder to obtain than phi and psi.
                This can be used when the heuristic is more complex than a function of the cf state.
                This is also used for LUCB, as several interventions are evaluated together. 
            dag (optional): a list of lists where each element i is the set of causal parents of V[i]

            args: V, U, D, F, u, psi, dag
        """
        self.V = V
        self.U = U
        self.D = D
        self.F = F
        self.u = u
        self.psi = psi
        self.dag = dag
        self.init_vars = dag[-1] if dag is not None else None
        self.sim = sim

    def apply_intervention(self, e):
        """
        Input: one intervention [(variable-value),...]
        Output: state
        """
        if self.F is not None:
            return self.F(self.u, e) # F returns the state
        return self.sim([e])[0][0] # sim returns [(state, phi, psi)]

    def apply_dict(self, d):
        """
        Input: one intervention {variable:value,...}
        Output: value of phi(e)
        """
        # Working function to try interventions manually
        e = [(self.V.index(var), value) for var, value in d.items()]
        return self.apply_intervention(e)
    
    def phi(self, e):
        """
        Input: one intervention [(variable-value),...]
        Output: value of phi(e)
        """
        if self.F is not None:
            return self.F(self.u, e)[-1] # F returns the state
        return self.sim([e])[0][1] # sim returns [(state, phi, psi)]
        
    def apply_interventions(self, E):
        """
        Input: list of interventions [[(variable-value),...],...]
        Output: list of (state, phi, psi)
        """
        if self.sim is not None: 
            return self.sim(E)
        out = []
        for e in E:
            s = self.F(self.u, e)
            out.append((s, s[-1], self.psi(s)))
        return out

    def get_v(self):
        return self.apply_intervention([])

    def get_input(self, base=True):
        if base:
            return self.get_input_beam_search()
        assert self.dag is not None
        return self.get_input_beam_search() | {"dag": self.dag, "init_var_ids":self.init_vars}

    def get_input_beam_search(self):
        return {"v": self.get_v()[:-1], "V": self.V[:-1], "D": self.D[:-1], "simulation": self.apply_interventions}
        