from actualcauses_local.lucb import lucb
import numpy as np
from collections import defaultdict

class SystemModel:
    def __init__(self, phi=None, psi=None):
        self.n_calls = 0
        self.psi = psi
        self.phi = phi

    def __call__(self, u:list, e: list[tuple]) -> list:
        raise NotImplementedError

    def evaluate_batch(self, u, E):
        raise NotImplementedError

class BaseNumpyModel(SystemModel):
    sub_N = 100_000 # used to chunk large batches
    def __init__(self, V, phi=None, psi=None, dtype=None):
        SystemModel.__init__(self, phi=phi, psi=psi)
        if dtype is None: self.dtype = bool
        else: self.dtype = dtype
        if self.phi is None: self.phi = lambda s: s[:,-1].astype(int)
        if self.psi is None: self.psi = lambda s: np.sum(s, axis=1) - 1
        self.dim2id = dict(zip(V, range(len(V))))
        self.reset_state()

    def reset_state(self):
        self.Es = None
        self.batch = None
        self.S = None

    def __getitem__(self, var: set):
        return self.S[:,self.dim2id[var]]
            
    def __setitem__(self, var: str, F_value):
        if self.batch: 
            self.S[:,self.dim2id[var]] = F_value
            if var in self.Es:
                for h_slice, value in self.Es[var]:
                    self.S[h_slice,self.dim2id[var]] = value
        else: 
            self.S[:,self.dim2id[var]] = self.Es.get(var, F_value)
            
    def __call__(self, u, e):
        self.batch = False
        self.S = np.zeros((1, len(self.dim2id)), dtype=self.dtype)
        self.Es = dict(e)
        self.simulate(u)
        S = self.S
        self.n_calls += S.shape[0]
        self.reset_state()
        return S.flatten().tolist()

    def evaluate_batch(self, u, E, N=1):
        self.batch = True
        out = []
        for i in range(0, len(E), self.sub_N):
            sub_E = E[i*self.sub_N:(i+1)*self.sub_N]
            self.S = np.zeros((len(sub_E)*N, len(self.dim2id)), dtype=self.dtype)
            self.Es = defaultdict(lambda: [])
            for i, e in enumerate(sub_E):
                for var, value in e:
                    self.Es[var].append((slice(i*N,(i+1)*N),value))
            self.simulate(u)
            out.append(np.array([self.phi(self.S), self.psi(self.S)]).T)
            self.n_calls += self.S.shape[0]
        if not len(out):
            return 
        return np.vstack(out)

class NoisyNumpyModel(BaseNumpyModel):
    def __init__(self, V, t, phi=None, psi=None, dtype=None):
        BaseNumpyModel.__init__(self, V, phi=phi, psi=psi, dtype=dtype)
        self.t = t

    def __setitem__(self, var, F_value):
        if self.batch: 
            self.S[:,self.dim2id[var]] = F_value
            if self.t > 0:
                ids = np.random.rand(self.S.shape[0]) < self.t
                self.S[ids,self.dim2id[var]] = 1 - self.S[ids,self.dim2id[var]]
            if var in self.Es:
                for h_slice, value in self.Es[var]:
                    self.S[h_slice,self.dim2id[var]] = value
        else: 
            if self.t > 0 and np.random.rand() < self.t: F_value = 1 - F_value
            self.S[:,self.dim2id[var]] = self.Es.get(var, F_value)

class AverageNumpyModel(NoisyNumpyModel):
    def __init__(self, V, t, N, phi=None, psi=None, dtype=None):
        NoisyNumpyModel.__init__(self, V, t, phi=phi, psi=psi, dtype=dtype)
        self.N = N

    def evaluate_batch(self, u, E, N=1):
        out = super().evaluate_batch(u, E, self.N)
        return out.reshape(len(E), self.N, 2).mean(axis=1)

class LUCBNumpyModel(NoisyNumpyModel):
    def __init__(self, V, t, lucb_params, phi=None, psi=None, dtype=None):
        NoisyNumpyModel.__init__(self, V, t, phi=phi, psi=psi, dtype=dtype)
        self.lucb_params = lucb_params

    def evaluate_batch(self, u, E, N=1):
        evaluator = lambda E, N: NoisyNumpyModel.evaluate_batch(self, u, E, N)
        return lucb(evaluator, E, **self.lucb_params)