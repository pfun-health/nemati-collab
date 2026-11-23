import os
import numpy as np

def load_stats(subject, database, path=None):
    """ load the stats dictionary from a .npy file.
    Input arguments
    subject (str) : The ID of the subject (e.g. "f1y01")
    database (str) : The name of the database (e.g. "fantasia")
    path [optional] (str) : The absolute path to the folder containing databases
        (default is /home/$USER/Public)
    Returns
    stats (dict) : dictionary containing the stats as described in the function compute_stats
    """
    subject_path = '{}.npy'.format(subject,)
    if not path:
        path = os.path.expanduser('~/Public')
    dpath = os.path.join(path, database)
    spath = os.path.join(dpath, subject_path)
    stats = np.load(spath)[()]
    return stats

def load_all_stats(database='fantasia', path=None):
    if not path:
        path = os.path.expanduser('~/Public')
    dpath = os.path.join(path, database)
    subjects = list(set([os.path.splitext(f)[0] for f in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, f))\
                         and '_' not in f]))
    all_stats = {}
    for subject in subjects:
        stats_loaded = load_stats(subject,database,path=path)
        all_stats.update({subject : stats_loaded})
    return all_stats
