import sys, os

sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

import pycardio
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt # import pyplot

def find_bp_hills(bp):
    mountains, hills = [], []
    diff_bp = np.diff(bp)
    np.insert(diff_bp,-1,diff_bp[-1])
    for ii, bpd in enumerate(diff_bp[:-5]):
        mountain, hill = None, None
        if any([any(h for h in hills if h >= ii), any(m for m in mountains if m >= ii)]):
            continue # Skip this iteration bc we have already looked at this BP cycle
        if bpd > 0 and diff_bp[ii+1] <= 0: # if this is a peak
            for ji, bpd0 in enumerate(diff_bp[ii+2:]): # look for the bottom of this hill/mountain
                if ji+ii+3 < len(diff_bp) and bpd0 < 0 and diff_bp[ji+ii+3] >= 0: # if this is the start of a hill/mountain
                    for ki, bpd1 in enumerate(diff_bp[ji+ii+4:]): # look for the next peak
                        if bpd1 > 0 and diff_bp[ki+ji+ii+5] <= 0: # this is the next peak
                            pk0 = ii+1 # first peak found
                            pk1 = ki+ji+ii+5 # second peak found
                            if bp[pk0] > bp[pk1]: mountain, hill = pk0, pk1
                            elif bp[pk1] > bp[pk0]: mountain, hill = pk1, pk0
                            if mountain is not None: mountains.append(mountain)
                            if hill is not None: hills.append(hill)
                            break
                    break
    return hills, mountains


def bp_kernel(x, a=1.6, b=0.5, c=0.3, d=1.9, e=0.0, f=0.1, g=0.5, h=2.0, I=1., J=1.): # defines the BP kernel with the parameters (a, b, c, d)
    return (I*np.exp(-np.pi*np.power(x, 2))+J*np.exp(-a*np.pi*np.power(c-x, 2))+g*np.exp(-b*np.pi*np.power(d-x,2)))*f*(h-x)+e

def fit_bp_kernel(bp_time, bp_cycle): # fit the BP kernel to a single BP cycle
    popt, pcov = optimize.curve_fit(bp_kernel, bp_time, bp_cycle,
                                    p0=[23., 10.9, -0.79, -0.65, 0.44, 0.03, .74, 9.0, 0.35, 1.2], # f2y04
                                    # p0=[24.4, 10.7, -0.80, -0.68, 0.33, 0.05, 1., 2., 1., 1.], # f2y02
                                    method='trf',
                                    maxfev=100000,
                                    bounds=([-10, -10, -10, -10, -1, -1, -1, 1, 0.01, 0.01],
                                            [100, 100, 10, 10, 1, 1, 1, 10, 10, 10]))
    print(popt)
    return popt

def convolve_bp(): # convolve the entire BP signal with the BP kernel
    pass


if __name__ == "__main__":

    database = 'fantasia' # Name of database

    ftype = 'ecg' # Type of ECG data

    path = os.path.expanduser('~/Public') # Path to database

    cycle_dict = {
        'f2y02': [9590, 170],
        'f2y03': [9650, 170],
        'f2y04': [9575, 150],
        'f2y05': [9660, 180],
        'f2y06': [9480, 220],
        'f2y07': [9430, 120],
        'f2y08': [9650, 220],
        'f2o01': [9480, 220],
        'f2o02': [9565, 220],
        'f2o03': [9920, 220],
        'f2o04': [9460, 170],
        'f2o05': [9520, 150],
        'f2o07': [9590, 140]
    }

    for subject_name in ['f2o07']:#['f2y{:02d}'.format(nr) for nr in range(2, 11)]:
        try:
            print(subject_name)

            # Load stats into `s`
            s = pycardio.load_data.load_stats(subject_name, database, path)

            bp = s['signals']['bp']
            time = s['signals']['time']
            resp = s['signals']['tv']
            time, bp, other_sigs = pycardio.process_data.DB.check_for_fucking_nans(time, bp, other_sigs=[resp,])
            resp = other_sigs[0]

            startx, endx = cycle_dict[subject_name]
            curve_time = np.linspace(-1, 1, endx)
            bpopt = fit_bp_kernel(curve_time, bp[startx:startx+endx])
            bp_curve = bp_kernel(curve_time, *bpopt)
            # bp_curve = bp_kernel(curve_time, a=18.0, b=31.0, c=-0.75,
                                 # d=-0.5, e=0.3, f=0.1, g=0.15)

            startx2 = startx
            endx2 = 5000
            bp2 = bp[startx2:startx2+endx2]
            time2 = time[startx2:startx2+endx2]
            resp2 = resp[startx2:startx2+endx2]

            bp_curve = pycardio.process_data.DB.normalize(bp_curve)
            time2, bp2_s = pycardio.process_data.DB.smooth_signal(time2, bp2, kernel=bp_curve)
            bp_hills, bp_mtns = find_bp_hills(bp2_s)

            
            # bp_pks = pycardio.process_data.DB.find_peaks(bp, method='max', order=200)
            # phase, bpi, bpi_dev = pycardio.process_data.DB.calc_RSA(time, bp_pks, resp)
            # phase = phase[:bpi_dev.size]

            ### Plot the BP deviation vs Respiratory phase
            
            plt.figure()
            plt.plot(time[startx:startx+endx], bp_curve, 'r--')
            plt.plot(time[startx:startx+endx], bp[startx:startx+endx], 'k')
            
            plt.figure()
            plt.plot(time2, bp2, 'k')
            plt.plot(time2, bp2_s[:len(time2)], 'b')
            # plt.plot(time2[bp_hills], bp2_s[bp_hills], 'bx')
            # plt.plot(time2[bp_mtns], bp2_s[bp_mtns], 'rx')
            # plt.savefig('./figs/hills_{}.svg'.format(subject_name,))

            # plt.plot(time[bp_pks], bp[bp_pks], 'rx')
            plt.show()
            
            # plt.figure() # new figure
            # plt.xlabel('respiratory phase (start of expiration = 0)')
            # plt.ylabel('BPI deviation (s)')
            # plt.title('BPI deviation vs Resp. cycle\nsubject: {}'.format(subject_name,))
            # plt.plot(phase, bpi_dev, 'rx') # plot phase on the x axis and rri_dev on the y axis (with a red x)
            # plt.xlim([0, 1]) # Set the x limit from 0 to 1
            # # plt.ylim([-0.3, 0.3]) # Set the y limit

            # Fit a polynomial

            # coeffs = np.polyfit(phase, bpi_dev, 3) # 3rd degree polynomial
            # p3 = np.poly1d(coeffs)
            # phase_order = np.sort(phase) # The phase sorted from min to max

            # plt.plot(phase_order, p3(phase_order), 'k') # Plot the fit as a black line

            # plt.savefig('./figs/bpi_{}'.format(subject_name,))
            # plt.show()
        except Exception as E:
            print(repr(E))
            sys.exit()
    
