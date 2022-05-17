
import configparser
import os
from edbo.plus.optimizer_botorch import EDBOplus
import pandas as pd
from pathlib import Path

# Load config file.
config = configparser.ConfigParser()
cwd = os.path.join(os.path.dirname(__file__))
config.read(f"{cwd}/config.ini")
cfg = config['EDBOPlus']
scratch_dir = Path(cfg['EDBO_DIR'])  # Path where we store the csv files.


def convert_web_dict_opt(web_dict_optimize):
    """
    Converts web dictionary to EDBO dictionary.

    Format of the input dictionary:
    web_dict_optimize = {
        'username': 'john@university.edu',
        'label': 'my_scope',
        'objectives_mode': 'yield, cost',
        'objectives_mode': 'max, min',
        'objectives_thresholds': '0.8, None',
        'columns_features': 'temperature, concentration',
        'batch': 5,
        }
    }
    """

    # Convert dictionary to lists.

    edbo_dict = {
        'scopename': web_dict_optimize['scopename'],
        'batch': int(web_dict_optimize['batch'])
    }

    # Convert string selected features to python list.
    selec_features = web_dict_optimize['selected_features'].replace(" ", "")
    selec_features = selec_features.split(',')
    edbo_dict.update({'columns_features': selec_features})

    obj = [v for key, v in web_dict_optimize.items() if key.startswith("obj")]
    maxmin = [v for key, v in web_dict_optimize.items() if key.startswith("maxormin")]
    thres = [v for key, v in web_dict_optimize.items() if key.startswith("threshold")]

    list_objectives = []
    list_maxmin = []
    list_thresholds = []

    # I need to separate the loops to make sure the input is consistent.
    for p in range(0, len(obj)):
        list_objectives.append(obj[p].replace(" ", ""))
    for p in range(0, len(maxmin)):
        list_maxmin.append(maxmin[p].replace(" ", ""))
    for p in range(0, len(thres)):
        list_thresholds.append(thres[p].replace(" ", ""))

    msg = 'Please fill out the objectives, objectives modes and thresholds.'
    assert len(list_thresholds) == len(list_maxmin) == len(list_objectives), msg

    # Check for None in thresholds (thresholds can be floats or None).
    for i in range(0, len(list_thresholds)):
        if list_thresholds[i] == 'None':
            list_thresholds[i] = None
        else:
            try:
                list_thresholds[i] = float(list_thresholds[i])
            except:
                list_thresholds[i] = None  # Overwrite and set to None.

    edbo_dict.update({'objectives': list_objectives})
    edbo_dict.update({'objectives_mode': list_maxmin})
    edbo_dict.update({'objective_thresholds': list_thresholds})
    return edbo_dict


def interface_optimize_scope(username, web_dict_optimize):

    edbo_dict = convert_web_dict_opt(web_dict_optimize)

    scopename = edbo_dict['scopename']

    filename = f"{scopename}.csv"
    filename_next = f"next_{scopename}.csv"
    filename_pred = f"pred_{scopename}.csv"
    path_scope = scratch_dir / username / scopename
    path_file = scratch_dir / username / scopename / filename
    path_next = scratch_dir / username / scopename / filename_next
    path_predictions = scratch_dir / username / scopename / filename_pred

    opt = EDBOplus()

    df = opt.run(
        directory=path_scope,
        filename=filename,
        objectives=edbo_dict['objectives'],
        objective_mode=edbo_dict['objectives_mode'],
        objective_thresholds=edbo_dict['objective_thresholds'],
        columns_features=edbo_dict['columns_features'],
        batch=edbo_dict['batch'],
    )

    next_df = df[df['priority'] == 1]
    next_df.to_csv(path_next, index=False)
    df.to_csv(path_file, index=False)

    return next_df, path_file, path_predictions


if __name__ == '__main__':  # Test example.
    user = 'user@uni.com'
    web_dict_opt = {
        'scopename': 'TEST_SCOPE',
        'obj1': 'yield',
        'obj2': 'cost',
        'maxormin1': 'max',
        'maxormin2': 'min',
        'selected_features': 'temperature, concentration',
        'threshold1': '0.8',
        'threshold2': 'None',
        'batch': '5',
        }
    from edbo.plus.backend.scope import interface_avail_features
    columns_available = interface_avail_features(username=user,
                                                 scopename='TEST_SCOPE')
    print('Columns available to show in web: \n', columns_available)
    df_after_opt = interface_optimize_scope(
        username=user,
        web_dict_optimize=web_dict_opt)
    print('Dataframe after optimization: \n\n', df_after_opt)

    columns_available = interface_avail_features(username=user,
                                                 scopename='TEST_SCOPE')
    print('Columns available to show in web:\n', columns_available)

    # Edit csv and add random entries.

    df1 = pd.read_csv(f"{cfg['EDBO_DIR']}{user}/TEST_SCOPE/TEST_SCOPE.csv")
    df1.loc[1, 'yield'] = 10
    df1.loc[1, 'cost'] = 30
    df1.loc[10, 'yield'] = 20
    df1.loc[10, 'cost'] = 100
    df1.loc[3, 'yield'] = 70
    df1.loc[3, 'cost'] = 10
    df1.to_csv(f"{cfg['EDBO_DIR']}{user}/TEST_SCOPE/TEST_SCOPE.csv", index=False)
    print(df1)
