import sys
import os
sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

import pycardio.process_data as pdat

if __name__ == "__main__":
    database = 'fantasia' # Name of database
    ftype = 'ecg' # Type of ECG data
    path = os.path.expanduser("~/Public") # Path to database
    excludes = [] # Subjects to exclude

    all_stats = pdat.load_all_stats(database=database, path=path) # Load stats from path

    # Get individual stats for each subject (in alphanumeric order of subject ID)
    subjects = pdat.get_stats_array(all_stats, "subject")
    phases = pdat.get_stats_array(all_stats, ["rsa", "resp_cycle"])
    rris = pdat.get_stats_array(all_stats, ["rsa", "rri"])
    beats = pdat.get_stats_array(all_stats, ["signals", "hb"])

    averages = pdat.get_averages(all_stats) # Print average values of certain stats

    import numpy as np
    import matplotlib.pyplot as plt
    import gc
    
    # DFA for a subject
    for sns in [sn for sn in list(subjects) if 'f1y' in sn]:
        ix = list(subjects).index(sns)
        subject = subjects[ix]
        phase = phases[ix]
        rri = rris[ix]
        beat_ix = beats[ix]
        nrange, F = pdat.DB.calc_DFA(beat_ix, rri, 4, 20)

        log_n = np.log10(nrange)
        log_F = np.log10(F)
    
        alpha = np.polyfit(log_n, log_F, 2)
        pa = np.poly1d(alpha)

        print(sns, alpha[1])
    
        plt.figure()
        plt.title("DFA for {}".format(subject))
        plt.plot(log_n, log_F, 'b')
        plt.plot(log_n, pa(log_n), 'r--')
        plt.xlabel("log_10( N )")
        plt.ylabel("log_10( F(N) )")
        plt.savefig('./figs/dfa_{}'.format(subject,))
        plt.close()
        gc.collect()
