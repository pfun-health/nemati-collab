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

    subject = 'f1o01'

    print('\ntest baro, ',subject)

    for b in np.linspace(0.00016, 0.0003, 5):

        print('baro: {:.5f}'.format(b))

        if subject == 'f2y03':

            # f2y03
            pdict = {"baro": b,
                     "d2": 0.642,
                     "d4": 0.527,
                     "hr": 0.00083,
                     "po": 0.15,
                     "rsa1": 0.000168, "rsa2": 0.00057,
                     "taup": 1367,
                     "rs1": 0.00001
            }

        elif subject == 'f1o01':

            # f1o01
            pdict = {"baro": b, # 0.00015
                     "d2": 0.616,
                     "d4": 0.587,
                     "hr": 0.000937,
                     "rsa1": 0.000129,
                     "rsa2": 0.000349,
                     "po": 0.01,
                     "rs1": 0.00001,
                     "taup": 1000}

        p = pycardio.setParams(dParams=None,
                               **pdict)


        data_stats = pycardio.load_data.load_stats(subject, 'fantasia', os.path.expanduser('~/Public'))

        data_time = data_stats['signals']['time']

        time_scale = 1000
        max_time = 1000 # seconds
        vlen = time_scale * max_time
        ndim = 12
        solution = np.zeros((vlen, ndim+1))
        end_time = ct.c_double(max_time)

        m.integrate(ct.c_int(vlen), ct.c_int(ndim), solution, end_time, ct.byref(p))

        tt = np.linspace(0, max_time, vlen)
        model_output = pycardio.ModelOutput(tt, solution,
                                            ix_offset=500,
                                            model_freq=time_scale)

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

        mhstr ='\nmodel rri {} '.format(rsa['rri_ave'])
        dhstr = ' data rri {}\n'.format(cvc_data['rri'].mean())

        mrstr = 'model insp {:f} +- {:f} sec '.format(aresp['insp_time'],np.std(aresp['insp_times']),) +\
                'model exp {:f} +- {:f} sec\n'.format(aresp['exp_time'],np.std(aresp['exp_times']),)

        drstr = 'data insp {:f} +- {:f} sec '.format(cvc_data['insp_time'],np.std(cvc_data['insp_times']),) +\
              'data exp {:f} +- {:f} sec\n'.format(cvc_data['exp_time'],np.std(cvc_data['exp_times']),)

        # CDF
        cdf_data = cvc_data['cdf']
        vec_data = cvc_data['beat_latency']
        data_rot = pycardio.calc_rot(cvc_data['rri'].mean(), cvc_data['exp_time'], cvc_data['insp_time'])    
        drot = 'data rot nr: {}\n'.format(data_rot)

        model_rot = pycardio.calc_rot(cvc['rri_ave'], cvc['exp_time'], cvc['insp_time'])
        mrot = 'model rot nr: {}\n'.format(model_rot)

        interp_vec, cdf_data, cdf_model = pycardio.utils.interp_cvc(vec_data, cdf_data, cvc_vec, cdf)

        cdiff = np.abs(cdf_data - cdf_model)
        mix = np.argmax(cdiff)
        D = cdiff.max()

        cmaxdiff = np.abs(cdf_data - interp_vec)
        bix = np.argmax(cmaxdiff)
        Dmax = cmaxdiff.max()
        
        print('D ', D)
        print('Dmax ', Dmax)

        print(mhstr, dhstr)
        print(mrstr, drstr)
        print(drot, mrot)

        with open(os.path.expanduser('~/public_html/figs/barostats.txt'),'a') as bf:
            bf.writelines(['\nbaro: {:.5f}\n'.format(b),
                           'D: {:.5f}\n'.format(D),
                           mhstr,dhstr,mrstr,drstr,
                           drot,mrot])


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
        plt.savefig(os.path.expanduser('~/public_html/figs/baro{:.5f}_example_cvc_{}.svg'.format(b,subject)))
        plt.close()

        plt.figure(figsize=(12,12),dpi=300)
        plt.title('P-P plot : {}'.format(subject))
        plt.plot(cdf_model, cdf_data, 'b.', label='(CDF data)/(CDF model)', ms=1.0)
        plt.plot(cdf_model, cdf_model, 'r--', label='1:1')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('CDF model')
        plt.ylabel('CDF data')
        plt.legend(loc='best')
        plt.savefig(os.path.expanduser('~/public_html/figs/baro{:.5f}_example_pp_{}.svg'.format(b,subject)))
        plt.close()

        plt.figure(figsize=(12,12),dpi=300)
        plt.title('RSA : {}'.format(subject))
        plt.plot(rphase, rri_dev, 'b.')
        plt.plot(cvc_data['resp_cycle'], cvc_data['rri_dev'], 'k')
        plt.fill_between(cvc_data['resp_cycle'], cvc_data['rri_dev']-np.sqrt(cvc_data['rsa_var']),
                         cvc_data['rri_dev']+np.sqrt(cvc_data['rsa_var']), facecolor='r', lw=0)
        plt.hlines(y=0, xmin=0, xmax=1, colors='k', linestyles='--', alpha=0.5)
        plt.ylabel('RRI Deviation (sec)')
        plt.xlabel('Phase [Expiration -> Expiration]')
        plt.xlim([0, 1])
        plt.ylim([-0.1, 0.1])
        plt.savefig(os.path.expanduser('~/public_html/figs/baro{:.5f}_example_rsa_{}.svg'.format(b,subject)))
        plt.close()

