import sys, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser("~/Git/nemati-collab")) # Path to library

import pycardio

import ctypes as ct
import numpy as np

if __name__ == '__main__':

    m = pycardio.setupModel()

    subject = 'f2y03'
    
    if len(sys.argv) > 1: # Give subject name as argument (e.g. 'f2y03')
        subject = sys.argv[1]

    if subject == 'f2y03':
        
        # f2y03 
        pdict = {"baro": 0.108,
                 "d2": 0.642,
                 "d4": 0.527,
                 "hr": 0.0008303,
                 "po": 0.01,
                 "rsa1": 0.000197, "rsa2": 0.0006,
                 "taup": 1367,
                 "rs1": 0.001 #0.00001
        }

    elif subject == 'f1o01':
    
        # f1o01
        pdict = {"baro": 0.0001, # 0.00015
                 "d2": 0.65,
                 "d4": 0.587,
                 "hr": 0.000937,
                 "rsa1": 0.0001,
                 "rsa2": 0.00035,
                 "po": 0.01,
                 "taup": 1500,
                 "rs1": 0.01}
    
    p = pycardio.setParams(dParams=None,
                           **pdict)


    data_stats = pycardio.load_data.load_stats(subject, 'fantasia', os.path.expanduser('~/Public'))

    data_time = data_stats['signals']['time']
    
    time_scale = 1000
    max_time = 6000 # seconds
    vlen = time_scale * max_time
    ndim = 12
    solution = np.zeros((vlen, ndim+1))
    end_time = ct.c_double(max_time)

    m.integrate(ct.c_int(vlen), ct.c_int(ndim), solution, end_time, ct.byref(p))

    tt = np.linspace(0, max_time, vlen)
    model_output = pycardio.ModelOutput(tt, solution,
                                        ix_offset=0,
                                        model_freq=time_scale,
                                        sampling_freq=time_scale)

    model_output.save_model("./model_{}.npy".format(subject,))
    
    aresp = model_output('ave_resp')

    sp = model_output.array
    tp = model_output.time

    rsa = model_output('rsa') 
    rri_dev = rsa['rri_dev']
    rphase = rsa['resp_cycle']

    hr_v = model_output('hr')

    cvc = model_output('cvc')
    cdf = cvc['cdf']
    cvc_vec = cvc['beat_latency']

    cvc_data = data_stats['cvc']
    cvc_data.update(data_stats['ave_resp'])
    cvc_data.update(data_stats['rsa'])

    print('model rri',rsa['rri_ave'])
    print('data rri',cvc_data['rri'].mean())

    print('model insp {:f} +- {:f} sec '.format(aresp['insp_time'],np.std(aresp['insp_times']),),
          'model exp {:f} +- {:f} sec'.format(aresp['exp_time'],np.std(aresp['exp_times']),) )
    
    print('data insp {:f} +- {:f} sec '.format(cvc_data['insp_time'],np.std(cvc_data['insp_times']),),
          'data exp {:f} +- {:f} sec'.format(cvc_data['exp_time'],np.std(cvc_data['exp_times']),) )
    
    cdf_data = cvc_data['cdf']
    vec_data = cvc_data['beat_latency']
    data_rot = pycardio.calc_rot(cvc_data['rri'].mean(), cvc_data['exp_time'], cvc_data['insp_time'])    
    print('data rot nr: ', data_rot)
    
    try:
        model_rot = pycardio.calc_rot(cvc['rri_ave'], cvc['exp_time'], cvc['insp_time'])
        print('model rot nr',np.floor(model_rot))

        interp_vec, cdf_data, cdf_model = pycardio.utils.interp_cvc(vec_data, cdf_data, cvc_vec, cdf)

        cdiff = np.abs(cdf_data - cdf_model)
        mix = np.argmax(cdiff)
        D = cdiff.max()

        cmaxdiff = np.abs(cdf_data - interp_vec)
        bix = np.argmax(cmaxdiff)
        Dmax = cmaxdiff.max()

        print('D ', D)
        print('Dmax ', Dmax)

        plt.figure(figsize=(12,12),dpi=300)
        plt.title('CDF : {}'.format(subject))
        plt.plot(interp_vec, cdf_model, 'b',label='model')
        plt.plot(interp_vec, cdf_data, 'k', label='data')
        # plt.plot(cvc_vec, cdf, 'g', label='model_raw')
        plt.plot(interp_vec, interp_vec, 'k--', alpha=0.5, label='bisector')
        plt.plot([interp_vec[bix], interp_vec[bix]], [interp_vec[bix], cdf_data[bix]], 'c-x', label='Dmax')
        plt.plot([interp_vec[mix], interp_vec[mix]], [cdf_model[mix], cdf_data[mix]], 'r-x', label='D')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Phase [Inspiration -> Inspiration]')
        plt.ylabel('CDF')
        plt.legend(loc='best')
        plt.savefig('figs/test_example_cvc_{}.svg'.format(subject))
        plt.close()

        plt.figure(figsize=(12,12),dpi=300)
        plt.title('RSA : {}'.format(subject))
        plt.plot(rphase, rri_dev, 'b.',label='model rsa')
        plt.plot(cvc_data['resp_cycle'], cvc_data['rri_dev'], 'k', label='data rsa')
        plt.fill_between(cvc_data['resp_cycle'], cvc_data['rri_dev']-np.sqrt(cvc_data['rsa_var']),
                         cvc_data['rri_dev']+np.sqrt(cvc_data['rsa_var']), facecolor='r', lw=0, label='data error')
        plt.hlines(y=0, xmin=0, xmax=1, colors='k', linestyles='--', alpha=0.5)
        plt.ylabel('RRI Deviation (sec)')
        plt.xlabel('Phase [Expiration -> Expiration]')
        plt.xlim([0, 1])
        plt.ylim([-0.1, 0.1])
        plt.legend(loc='best')
        plt.savefig('figs/test_example_rsa_{}.svg'.format(subject))
        plt.close()

    except (OverflowError, ValueError, IndexError):
        print('Failed to do CVC, RSA. Error (rri is small probably).')

    finally:
        # Neurons
        plt.figure(figsize=(12,12), dpi=300)
        plt.subplot(511)
        plt.plot(tp, sp[:,0])
        plt.ylabel('pre-i/i')

        plt.subplot(512)
        plt.plot(tp, sp[:,1])
        plt.ylabel('early-i')

        plt.subplot(513)
        plt.plot(tp, sp[:,2])
        plt.ylabel('aug-e')

        plt.subplot(514)
        plt.plot(tp, sp[:,3])
        plt.ylabel('post-i')

        plt.subplot(515)
        plt.plot(tp, sp[:,4])
        plt.ylabel('ramp-i')
        plt.savefig('figs/test_example_neurons_{}.svg'.format(subject))
        plt.close()
        
        # Model output (signals)
        plt.figure(figsize=(12,12),dpi=300)
        plt.subplot(411)
        plt.plot(tp, sp[:,10]) # heart rate rate
        plt.ylabel('heart phase')

        plt.subplot(412)
        # plt.plot(tp, model_output('bp_s')) # smoothed blood pressure
        plt.plot(tp, model_output('bp'))
        plt.ylabel("blood pressure")

        plt.subplot(413)
        plt.plot(tp, model_output('tv')) # tidal volume
        plt.ylabel("tidal volume")

        plt.subplot(414)
        plt.plot(tp[hr_v[:,0].astype('int')], hr_v[:,1]) # heart rate
        plt.ylabel('Heart Rate (bps)')
        plt.savefig('figs/test_example_output_{}.svg'.format(subject))
        plt.close()
