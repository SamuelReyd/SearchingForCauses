from collections.abc import Iterable
import time
from .mbs import beam_search, show_rules
from .isi import iterative_identification
from .system_model import SystemModel
from typing import Any

class SCM:
    def __init__(self, V:list[str], U:list[str], D:list[list], u:list,
                 model:SystemModel, dag:list[list]=None, v:list=None, ):
        """
        SCM object that includes everything needed to execute the algorithms
            V: list of endogenous variables labels (the target must be the last variable)
            U: list of exogenous variables labels
            D: list of discrete domains for the variables in V (in the same order)
            u: context (values of U in the right order)
            model (optional): a SystemModel object that must implement functions apply and evaluate_batch 
            dag (optional): a list of lists where each element i is the set of causal parents of V[i].
            v (optional): a list of values for v. If none is provided, then F(u,[]) is used. 
                Providing v can be useful for stochastic experiments.

            args: V, U, D, u, dag
        """
        # Mandatory parameters
        self.V = V
        self.U = U
        self.model = model
        if isinstance(D[0], Iterable):
            self.D = D
        else:
            self.D = [D] * len(V)
        self.u = u
        if v is None:
            self.v = self({})
        else:
            self.v = v
            
        # Interventions and evaluation functions
        self.dag = dag
        self.init_vars = dag[V[-1]] if dag is not None else None
        
        self.causes = None
        self.causes_hashable = None
        self.witnesses = None
        self.identification_output = None
        self.interventions = None
        self.identification_time = None
        self.n_calls = None
    
    def __call__(self, e:list[tuple[str,Any]]) -> list[Any]:
        return self.model(self.u, e)
        
    def evaluate_batch(self, E: list[list[tuple[str,Any]]], N:int=1) -> list[tuple[float,float]]:
        return self.model.evaluate_batch(self.u, E, N)

    def get_input(self, base=True):
        if base:
            return self.get_input_beam_search()
        assert self.dag is not None
        return self.get_input_beam_search() | {"dag": self.dag, "PA_T":self.init_vars}

    def get_input_beam_search(self):
        return {"v": self.v[:-1], "V": self.V[:-1], "D": self.D[:-1], "simulation": self.evaluate_batch}

    def find_causes(self, ISI=False, **kwargs):
        t = time.time()
        self.model.n_calls = 0
        if ISI:
            out = iterative_identification(**self.get_input(False), **kwargs)
        else:
            out = beam_search(**self.get_input(), **kwargs)
        self.identification_time = time.time() - t
        self.identification_output = out
        self.causes = [elt[3] for elt in out]
        self.causes_hashable = [tuple(sorted(elt)) for elt in self.causes]
        self.witnesses = [elt[4] for elt in out]
        self.interventions = [elt[0] for elt in out]
        self.n_calls = self.model.n_calls
        
    def show_indentification_result(self):
        print(f"Found {len(self.causes)} causes in {self.identification_time:.3f}s with {self.n_calls} model calls\n")
        show_rules(self.identification_output)