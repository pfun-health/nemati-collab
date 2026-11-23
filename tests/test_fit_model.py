import sys, os

sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

import pycardio

if __name__ == "__main__":

    database = 'fantasia' # Name of database

    ftype = 'ecg' # Type of ECG data

    path = os.path.expanduser('~/Public') # Path to database

    subject_name = "f1o07" # name of the subject

    
    # Load stats into `s`
    s = pycardio.load_data.load_stats(subject_name, database, path)
    
    ### Get the RSA stats
    rsa = s['rsa']
    phase = rsa['resp_cycle'] # Get the respiratory cycle (normalized as phase 0 to 1)
    rri_dev = rsa['rri_dev'] # Get the RRI deviation (in seconds)
    rri = rsa['rri']
    phase = phase[:len(rri_dev)]
    
    cost = pycardio.fit_model.CF_simple(rri, rri)
    print("should be 0: {}".format(cost,))

    s2 = pycardio.load_data.load_stats('f1y03', database, path)
    rri2 = s2['rsa']['rri']
    cost2 = pycardio.fit_model.CF_simple(rri.mean(), rri2.mean(), rri2.var())
    print("should be not 0: {}".format(cost2,))

    
    # ### Plot the RRI deviation vs Respiratory phase
    # import matplotlib.pyplot as plt # import pyplot
    
    # plt.figure() # new figure

    # plt.xlabel('respiratory phase (start of expiration = 0)')
    # plt.ylabel('RRI deviation (s)')
    # plt.title('RRI deviation vs Resp. cycle\nsubject: {}'.format(subject_name,))

    # plt.plot(phase, rri_dev, 'rx') # plot phase on the x axis and rri_dev on the y axis (with a red x)
    # plt.xlim([0, 1]) # Set the x limit from 0 to 1
    # plt.ylim([-0.3, 0.3]) # Set the y limit

    # # Fit a polynomial
    # import numpy as np
    # coeffs = np.polyfit(phase, rri_dev, 3) # 3rd degree polynomial
    # p3 = np.poly1d(coeffs)
    # phase_order = np.sort(phase) # The phase sorted from min to max

    # plt.plot(phase_order, p3(phase_order), 'k') # Plot the fit as a black line
    
    # plt.show()

    
