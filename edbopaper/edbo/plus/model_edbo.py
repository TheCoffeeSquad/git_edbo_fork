
from edbo.models import GP_Model
import numpy as np
from gpytorch.priors import GammaPrior
from edbo.base_models import fast_computation


def build_and_optimize_model(train_x, train_y, mordred=False,
                             ohe_features=False):

    fast_computation(True)
    n_features = np.shape(train_x)[1]

    train_y = train_y.squeeze(-1)

    if n_features < 5:
        lengthscale_prior = [GammaPrior(1.2, 1.1), 0.2]
        outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
        noise_prior = [GammaPrior(1.05, 0.5), 0.1]
    # DFT optimized priors
    elif mordred and n_features < 100:
        lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
        outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
        noise_prior = [GammaPrior(1.5, 0.1), 5.0]
        # Mordred optimized priors
    elif mordred:
        lengthscale_prior = [GammaPrior(2.0, 0.1), 10.0]
        outputscale_prior = [GammaPrior(2.0, 0.1), 10.0]
        noise_prior = [GammaPrior(1.5, 0.1), 5.0]
    # OHE optimized priors
    else:
        lengthscale_prior = [GammaPrior(3.0, 1.0), 2.0]
        outputscale_prior = [GammaPrior(5.0, 0.2), 20.0]
        noise_prior = [GammaPrior(1.5, 0.1), 5.0]

    gp = GP_Model(X=train_x, y=train_y,
                  lengthscale_prior=lengthscale_prior,
                  outputscale_prior=outputscale_prior,
                  noise_prior=noise_prior, #n_restarts=1,
                  )
    gp.fit()

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
    # noise = gp.likelihood.noise.item()
    #
    # df_row = [model_number, len(train_y), noise]
    #
    # for ls in range(0, n_features):  # Lengthscales
    #     ls_i = gp.model.covar_module.base_kernel.lengthscale[0][ls].item()
    #     df_row.append(ls_i)
    #
    # dict = {}
    # for i in range(0, len(columns_name)):
    #     dict[columns_name[i]] = df_row[i]
    # df = df.append(dict, ignore_index=True)
    # df.to_csv(filename_hyper_opt, index=False)

    return gp.model, gp.likelihood