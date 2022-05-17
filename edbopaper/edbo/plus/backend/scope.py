
import numpy as np
import pandas as pd

from edbo.plus.scope_generator import create_reaction_scope
import os
import shutil
import configparser
from pathlib import Path


# Load config file.
config = configparser.ConfigParser()
cwd = os.path.join(os.path.dirname(__file__))
config.read(f"{cwd}/config.ini")
cfg = config['EDBOPlus']
scratch_dir = Path(cfg['EDBO_DIR'])  # Path where we store the csv files.


def check_number_scopes_user(username):
    """
    Check number of scopes that have been stored by a given user.
    """
    msg = "Maximum number of scopes per user reached. Please delete "
    msg += "any of your scopes in order to create a new one."

    if os.path.exists(scratch_dir / username):
        assert len(interface_get_available_scopes(username)) < \
               int(cfg['MAX_NUMBER_SCOPES']), msg


def interface_delete_scope(username, scopename):
    """
    Removes scope from directory.
    """
    path_scope = scratch_dir / username / scopename
    if os.path.exists(path_scope):
        shutil.rmtree(path_scope)


def convert_web_dict_scope(web_dict):
    """ Function to convert from the HTML dictionary to a EDBO dictionary."""

    dict_edbo = {'scopename': web_dict['scopename'],
                 'components': {}}

    feat = [value for key, value in web_dict.items() if key.startswith("feat")]
    vals = [value for key, value in web_dict.items() if key.startswith("val")]

    for p in range(0, len(feat)):
        f, v = feat[p], vals[p]
        clean_vals = v.replace(" ", "")  # Clean whitespaces.
        val = clean_vals.split(",")  # Split values into a list.
        # Try to convert values in list to floats (otherwise keep them as str).
        try:
            values = list(np.float_(val))
        except ValueError:
            values = val
        dict_edbo['components'].update({f: values})
    return dict_edbo


def create_scope(username, dict_edbo):
    """
    Generates reaction scope from a web dictionary.

    Format of the input dictionary:

    dict_edbo = {
        'username': 'john@university.edu',
        'scopename': 'my_scope',
        'components': {
            'Solvent': ['Water', 'DMSO'],
            'Ligand': ['L1', 'L2'],
            'Concentration': [0.1, 0.2, 0.3],
            'Temperature': [50, 60, 70]
        }
    }


    """

    scopename = dict_edbo['scopename']
    comp = dict_edbo['components']

    # Check user is not exceeding the number of reaction scopes generated.
    check_number_scopes_user(username)

    # Check reaction scope is not too big.
    keys = comp.keys()

    n = []
    [n.append(len(comp[key])) for key in keys]

    # Check number of reactions in the scope is not huge.
    msg = "The number of combinations in the scope has been exceeded. \n"
    msg += "At the moment we only accept scopes with less than"
    msg += f" {cfg['SCOPE_SIZE_LIMIT']} combinations."
    assert np.prod(n) < int(cfg['SCOPE_SIZE_LIMIT']), msg

    # Get paths to files and folders.
    filename = f"{scopename}.csv"
    path_to_scope = (scratch_dir / username / scopename)
    path_to_file = path_to_scope.joinpath(filename)

    # Do not overwrite scopes, instead allow the user to rename or delete.
    msg = "Scope with the same scope name name already exists. \n "
    msg += "Please select another name or delete the previous scope."
    assert os.path.exists(path_to_file) is False, msg

    try:
        os.makedirs(path_to_scope)

        df = create_reaction_scope(components=comp,
                                   directory=path_to_scope,
                                   filename=filename,
                                   check_overwrite=False)
        return df
    except:
        os.rmdir(path_to_scope)


def interface_get_available_scopes(username):
    """
    Returns a list of scopes for a given user.
    """
    try:
        path = scratch_dir / username
        subfolders = [os.path.join(path, f) for f in os.listdir(path) if
                      os.path.isdir(os.path.join(path, f))]
        r = []
        [r.append(sf.split('/')[-1]) for sf in subfolders]
        return r

    except:
        return []


def interface_upload_scope(username, scopename, df):
    """
    Function to upload csv file to the username scratch directory.

    Input:
    username: string with the username
    scopename: name of the scope
    df: df in Pandas format.

    """
    # Check whether df is larger than accepted.
    msg = f"Scope size is too big. Maximum number of allowed combinations is: {int(cfg['SCOPE_SIZE_LIMIT'])}"
    assert len(df.values) < int(cfg['SCOPE_SIZE_LIMIT']), msg

    msg = f"Too many features. Maximum number of allowed features: {int(cfg['SCOPE_SIZE_LIMIT'])}"
    assert len(df.columns.values) < int(cfg['MAX_NUMBER_OF_FEATURES']), msg

    # Check format.
    try:
        filename = f"{scopename}.csv"
        path_scope = scratch_dir / username / scopename
        path_file = path_scope / filename

        if not os.path.exists(path_scope):
            os.makedirs(path_scope)

        df.to_csv(path_file, index=False)
        msg = 'Scope was succesfully uploaded'
        return msg
    except:
        msg = 'Scope was not uploaded. Please check the format of the scope.'
        return msg


def interface_get_scope_upload_path(username, scopename):
    """
    Function to upload csv file to the username scratch directory.

    Input:
    username: string with the username
    scopename: name of the scope
    Returns: path to store the scope.
    """
    filename = f"{scopename}.csv"
    filename_pred = f"pred_{scopename}.csv"
    path_scope = scratch_dir / username / scopename
    path_file = path_scope / filename
    path_pred = path_scope / filename_pred

    if not os.path.exists(path_scope):
        os.makedirs(path_scope)

    return path_file


def interface_get_scope(username, scopename):
    """
    Pass username and scopename and returns the df for the scope.
    """
    filename = f"{scopename}.csv"
    path_file = scratch_dir / username / scopename / filename

    msg = 'Scope not available'
    assert os.path.exists(path_file), msg

    df = pd.read_csv(f"{path_file}")
    df = df.dropna(axis='columns', how='all')
    return df


def interface_avail_features(username, scopename):
    """
    Returns a list of strings with the names of the features available in
    the dataframe. This allows the user to select specific columns to build
    the regression model.
    By default, if not specified, it will select all the columns (train_x)
    except the objectives (train_y).
    """
    filename = f"{scopename}.csv"
    path_file = scratch_dir / username / scopename / filename
    df = pd.read_csv(f"{path_file}")
    df = df.dropna(axis='columns', how='all')

    all_columns = df.columns.values

    features_columns = []
    for column in all_columns:
        if column != 'priority' and not np.any(df[column].values == 'PENDING'):
            features_columns.append(column)
    return features_columns


def interface_create_scope(username, web_dict):
    """Interface between the create scope in HTML front-end and EDBO"""
    edbo_dict = convert_web_dict_scope(web_dict)
    df = create_scope(username, edbo_dict)
    return df


if __name__ == '__main__':  # Test example.

    user = 'user@uni.com'

    web_d = {
        "feat1": "temperature",
        "feat2": "concentration",
        "feat3": "solvent",
        "scopename": "TEST_SCOPE",
        "val1": "20, 30, 40, 50, 100",
        "val2": "0.1, 0.3,0.5",
        "val3": "DMSO, THF"
    }

    print(interface_get_available_scopes(username=user))
    interface_delete_scope(username=user, scopename='TEST_SCOPE')
    df_web = interface_create_scope(username=user,
                                    web_dict=web_d)
    print(interface_get_available_scopes(username=user))
    print(df_web)

    df_web.loc[0, 'solvent'] = 'My_solvent'
    print('Scope after introducing one entry:\n', df_web)

    print('Scope pulled from drive (before uploading):\n',
          interface_get_scope(user, scopename=web_d['scopename']))

    print('Uploading scope to drive:\n',
          interface_upload_scope(username=user,
                                 scopename=web_d['scopename'],
                                 df=df_web))

    print('Scope pulled from drive (after uploading):\n',
          interface_get_scope(user, scopename=web_d['scopename']))
