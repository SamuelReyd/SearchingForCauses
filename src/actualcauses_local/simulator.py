class SystemModel:
    def __init__(self, psi=None, phi=None):
        self.n_call = 0
        self.psi = psi
        self.phi = phi

    def apply(self, u:list, e: list[tuple]) -> list:
        raise NotImplementedError

    def evaluate_batch(self, u, E):
        raise NotImplementedError

class BaseNumpyModel(SystemModel):
    sub_N = 100_000 # used to chunk large batches
    def __init__(self, V, phi=None, psi=None):
        if phi is None: self.phi = lambda s: s[-1]
        else: self.phi = phi
        if psi is None: self.psi = lambda s: np.sum(s) - 1
        else: self.phi = phi
        self.dim2id = dict(zip(V, range(len(V))))
        self.reset_state()

    def reset_state(self):
        self.Es = None
        self.batch = None
        self.S = None

    def __getitem__(self, var: set):
        return self.S[:,self.dim2id[var]]
            
    def __setitem__(self, var: str, F_value):
        # if self.t > 0:
        #     ids = np.random.rand(self.S.shape[0]) < self.t
        #     self.S[ids,self.dim2id[var]] = 1 - self.S[ids,self.dim2id[var]]
        if self.batch: 
            self.S[:,self.dim2id[var]] = F_value
            if var in self.Es:
                for h_slice, value in self.Es[var]:
                    self.S[h_slice,self.dim2id[var]] = value
        else: 
            self.S[:,self.dim2id[var]] = self.Es.get(var, F_value)
            
    def apply(self, u, e):
        self.batch = False
        self.S = np.array((1, len(self.dim2id)), dtype=bool)
        self.Es = dict(e)
        self.simulate(u)
        S = self.S
        self.n_calls += S.shape[0]
        self.reset_state()
        return S.flatten()

    def evaluate_batch(self, u, E, N=1):
        self.batch = True
        out = []
        for i in range(0, len(E), self.sub_N):
            sub_E = E[i*sub_N:(i+1)*self.sub_N]
            self.S = np.zeros((len(sub_E)*N, len(self.dim2id)), dtype=bool)
            self.Es = defaultdict(lambda: [])
            for i, e in enumerate(sub_E):
                for var, value in e:
                    self.Es[var].append((slice(i*N,(i+1)*N),value))
            self.simulate(u)
            out.append(np.array(self.phi(self.S), self.psi(self.S)).T)
            self.n_calls += self.S.shape[0]
        return np.vstack(out)