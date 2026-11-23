import sys, os
sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

import pycardio

from pycardio.process_data import *

if __name__ == '__main__':
    database = 'fantasia'
    ftype = 'ecg'
    path = os.path.expanduser("~/Public")

    excludes = []
    
    if len(sys.argv) > 1:

        erange = [ii for ii in range(1, 11)]
        for group in [1, 2]:
            for age in ['o', 'y']:
                excludes = exclude_range(excludes, group, age, erange)

        for subject_name in sys.argv[1:]:
            print(subject_name)
            excludes.remove(subject_name)

        # print(excludes)

    # dat is a database object
    dat = compute_all_stats(database, path=path,
                            save=True, overwrite=True,
                            exclude=excludes)
    
    computed_stats = dict(dat.stats) # Dictionary of all stats, keys are subjects
    averages = get_averages(computed_stats) # Get certain average values and print them
