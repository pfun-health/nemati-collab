import sys
import os
sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

import pycardio.process_data as pdat

if __name__ == "__main__":
    database = 'fantasia' # Name of database
    ftype = 'ecg' # Type of ECG data
    path = os.path.expanduser("~/Public") # Path to database
    excludes = [] # Subjects to exclude

    # dat is a database object
    # dat = pdat.compute_all_stats(database, path=path, save=False, overwrite=False, exclude=excludes)

    all_stats = pdat.load_all_stats(database=database, path=path) # Load stats from path

    # Get individual stats for each subject (in alphanumeric order of subject ID)
    subjects = pdat.get_stats_array(all_stats, "subject")
    mean_hr = pdat.get_stats_array(all_stats, ["descriptive", "hr"])
    mean_resp = pdat.get_stats_array(all_stats, ["descriptive", "resp_rate"])
    rot_nr = pdat.get_stats_array(all_stats, ["descriptive", "rot_nr"])
    ave_exp = pdat.get_stats_array(all_stats, ["ave_resp", "exp_time"])
    ave_insp = pdat.get_stats_array(all_stats, ["ave_resp", "insp_time"])

    averages = pdat.get_averages(all_stats) # Print average values of certain stats

    # Get all the RRI stats for all subjects
    rri_devs = pdat.get_stats_array(all_stats, ["rsa", "rri_dev"])
    phases = pdat.get_stats_array(all_stats, ["rsa", "resp_cycle"])

    # Get data from a single subject
    index = 12
    subject = subjects[index] # The name of the subject
    rri_dev = rri_devs[index] # The RRI-deviation from the subject
    phase = phases[index] # The respiratory phase data series from the subject

    rri_dev = rri_dev[rri_dev != -999] # remove flagged values
    phase = phase[phase != -999]
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(phase[:len(rri_dev)], rri_dev, 'rx')
    plt.xlim([0, 1])
    
    plt.xlabel('respiratory phase (start of inspiration = 0)')
    plt.ylabel('RRI deviation (s)')
    plt.title('RRI deviation vs Resp. cycle\nsubject: {}'.format(subject,))
    plt.show()

    
