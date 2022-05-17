
import torch
import gpytorch
import numpy as np

def to_torch(data, gpu=False):
    """
    Convert from pandas dataframe or numpy array to torch array.
    """

    if 'torch' in str(type(data)):
        torch_data = data

    else:
        try:
            torch_data = torch.from_numpy \
                (np.array(data).astype('float')).float()
        except:
            torch_data = torch.tensor(data).float()

    if torch.cuda.is_available() and gpu == True:
        torch_data = torch_data.cuda()

    return torch_data.double()


def fast_computation(fastQ):
    """Function for turning on/off GPyTorch fast computation features.

    Parameters
    ----------
    fastQ : bool
        If True, gpytorch fast computation features will be utilized. Modified
        settings include: (1) fast_pred_var, (2) fast_pred_samples,
        (3) covar_root_decomposition, (4) log_prob, (5) solves, (6) deterministic_probes,
        (7) memory_efficient.

    Returns
    ----------
    None
    """

    gpytorch.settings.fast_pred_var._state = fastQ
    gpytorch.settings.fast_pred_samples._state = fastQ
    gpytorch.settings.fast_computations.covar_root_decomposition._state = fastQ
    gpytorch.settings.fast_computations.log_prob._state = fastQ
    gpytorch.settings.fast_computations.solves._state = fastQ
    gpytorch.settings.deterministic_probes._state = fastQ
    gpytorch.settings.memory_efficient._state = fastQ


class EDBOStandardScaler:
    def __init__(self):
        pass

    def fit(self, x):
        self.mu  = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x):
        for obj in range(0, len(self.std)):
            if self.std[obj] == 0.0:
                self.std[obj] = 1e-6
        return (x-[self.mu])/[self.std]

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        for obj in range(0, len(self.std)):
            if self.std[obj] == 0.0:
                self.std[obj] = 1e-6
        return (x-[self.mu])/[self.std]

    def inverse_transform(self, x):
        '''To transform mean'''
        return x * [self.std] + [self.mu]

    def inverse_transform_var(self, x):
        '''To transform standard deviation'''
        return x * [self.std]
