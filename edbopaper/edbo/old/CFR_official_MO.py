import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


sys.path.append('/home/sanjay/AFRL/git_edbo/edbopaper')

from plus.optimizer_botorch import EDBOplus

#setting up reaction components
#np.arange creates a list of numbers from (start,stop,stepsize)
components = {
              'temperature':np.arange(30,140,3).tolist(),   # Discrete grid of concentrations
              'time': np.arange(1,45,2).tolist(),
              'stoichiometry': np.arange(0.33,0.66,0.025).tolist()}


#need to generage the data scope first
# scope = EDBOplus.generate_reaction_scope(components=components)

#Initialize EDBOplus class that will store predicted means, variances, etc.
bo = EDBOplus()

args = {'objectives': ['production_rate_(g/hr)','yield'], 'objective_mode': ['max','max'], 'batch': 3, 'seed': 42}


#Run this cell on scope data (csv file called reaction.csv)
bo.run(**args)




