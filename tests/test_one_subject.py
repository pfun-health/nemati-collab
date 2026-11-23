import sys, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # import pyplot
    

sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

import pycardio

import numpy as np

if __name__ == "__main__":

    database = 'fantasia' # Name of database

    ftype = 'ecg' # Type of ECG data

    path = os.path.expanduser('~/Public') # Path to database

    subject_name = "f1o01" # name of the subject

    if len(sys.argv) > 1:
        for subject_name in sys.argv[1:]:
        
            # Load stats into `s`
            s = pycardio.load_data.load_stats(subject_name, database, path)


            ### RSA
            rsa = s['rsa']
            phase = rsa['resp_cycle'] # Get the respiratory cycle (normalized as phase 0 to 1)
            rri_dev = rsa['rri_dev'] # Get the RRI deviation (in seconds)

            plt.figure(figsize=(12,12),dpi=300)

            plt.hlines(y=0, xmin=0, xmax=1, colors='k', linestyles='--', lw=0.5, alpha=0.5)

            plt.xlabel('respiratory phase (start of expiration = 0)')
            plt.ylabel('RRI deviation (s)')
            plt.title('RRI deviation vs Resp. cycle\nsubject: {}'.format(subject_name,))

            plt.plot(phase, rri_dev, 'k.') # plot phase on the x axis and rri_dev on the y axis (with a red x)
            # print(np.sqrt(rsa['rsa_var'][0]))
            plt.fill_between(phase, rri_dev-np.sqrt(rsa['rsa_var']), rri_dev+np.sqrt(rsa['rsa_var']), facecolor='r', lw=0)
            plt.ylim([-0.1, 0.1])
            plt.xlim([0, 1]) # Set the x limit from 0 to 1
            plt.savefig('figs/example_rsa_{}.svg'.format(subject_name,))

            # CVC
            cvc_data = s['cvc']
            cvc_data.update(s['ave_resp'])
            cvc_data.update(s['rsa'])

            data_rot = pycardio.calc_rot(cvc_data['rri'].mean(), cvc_data['exp_time'], cvc_data['insp_time'])

            print('data',data_rot)

            plt.figure(figsize=(12,12),dpi=300)
            plt.xlabel('beat phase latency')
            plt.ylabel('CDF')
            plt.title('CVC, subject: {}'.format(subject_name,))

            plt.plot(cvc_data['beat_latency'], cvc_data['cdf'], 'k.', label='cdf')
            plt.xlim([0, 1])

            plt.legend(loc='best')



            plt.savefig('figs/example_cvc_{}.svg'.format(subject_name,))
 
