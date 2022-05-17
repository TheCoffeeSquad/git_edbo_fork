
import torch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
import numpy as np
from .utils import to_torch
# from .utils import fast_computation

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

def build_and_optimize_model(train_x, train_y, ohe_features=False,
                             fast_computations=True):
    """ Builds model and optimizes it."""

    # fast_computation(True)

    print('Using hyperparameters optimized for continuous variables.')
    gp_options = {
        'ls_prior1': 2.0, 'ls_prior2': 0.2, 'ls_prior3': 5.0,
        'out_prior1': 5.0, 'out_prior2': 0.5, 'out_prior3': 8.0,
        'noise_prior1': 1.5, 'noise_prior2': 0.1, 'noise_prior3': 5.0,
        'noise_constraint': 1e-5,
    }


    n_features = np.shape(train_x)[1]
    # print('Building and optimizing ML model...')

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y,
                                               likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            kernels = MaternKernel(
                ard_num_dims=n_features,
                lengthscale_prior=GammaPrior(gp_options['ls_prior1'],
                                             gp_options['ls_prior2'])
            )

            self.covar_module = ScaleKernel(
                kernels,
                outputscale_prior=GammaPrior(gp_options['out_prior1'],
                                             gp_options['out_prior2']))
            try:
                ls_init = to_torch(gp_options['ls_prior3']).double()
                self.covar_module.base_kernel.lengthscale = ls_init
            except:
                uniform = to_torch(gp_options['ls_prior3']).double()
                ls_init = torch.ones(n_features) * uniform
                self.covar_module.base_kernel.lengthscale = ls_init

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        GammaPrior(gp_options['noise_prior1'], gp_options['noise_prior2'])
    )

    likelihood.noise = to_torch(gp_options['noise_prior3']).double()

    model = ExactGPModel(train_x, train_y, likelihood)

    model.likelihood.noise_covar.register_constraint(
        "raw_noise", GreaterThan(gp_options['noise_constraint'])
    )

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 1000
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.squeeze(-1))
        loss.backward()
        optimizer.step()

    # # Track hyperparameter convergence w.r.t. optimizer iterations.
    # import os
    # import pandas as pd
    # filename_hyper_opt = 'track_opt_hyperparameters.csv'
    # columns_name = ['model_number', 'samples', 'noise']
    # for i in range(0, n_features):
    #     columns_name.append(f"lenghscale_{i}")
    # if not os.path.exists(filename_hyper_opt):
    #     df = pd.DataFrame(columns=columns_name)
    # else:
    #     df = pd.read_csv(filename_hyper_opt)
    #
    # model_number = len(df['samples'][df['samples'].isin([len(train_y)])])  # Model number.
    # noise = model.likelihood.noise.item()
    #
    # df_row = [model_number, len(train_y), noise]
    #
    # for ls in range(0, n_features):  # Lengthscales
    #     ls_i = model.covar_module.base_kernel.lengthscale[0][ls].item()
    #     df_row.append(ls_i)
    #
    # dict = {}
    # for i in range(0, len(columns_name)):
    #     dict[columns_name[i]] = df_row[i]
    # df = df.append(dict, ignore_index=True)
    # df.to_csv(filename_hyper_opt, index=False)

    model.eval()
    likelihood.eval()
    return model, likelihood  # Optimized model

