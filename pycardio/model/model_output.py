from ..process_data.db import DB
import numpy as np

class ModelOutput(object):
    """
    Map the array outputted from the model to a dict\
    organized like the stats dict in the process_data module.
    """
    def __init__(self, time, model_output_array,
                sampling_freq=250, model_freq=1000,
                 ix_offset=0):
        self.output = { "cvc" : {}, "rsa" : {} }
        self.downsample_step = int(model_freq/sampling_freq)
        try:
            self.time = time[ix_offset::self.downsample_step]
            self.array = model_output_array[ix_offset::self.downsample_step,:] # downsample model to match data sampling rate
        except ValueError:
            print('WARNING: downsample failed!')
            self.time = time[ix_offset:]
            self.array = model_output_array[ix_offset:,:]
        finally:
            self._setupOutput()

    def __call__(self, output_key=None):
        if not output_key: # return the modified model_output_array 
            return self.array
        if output_key not in self.output.keys():
            self.output[output_key] = {} # instantiate dict for output  
        if output_key in ['rsa', 'rri', 'rri_dev', 'rri_ave', 'cvc']:
            rri_times, resp_cycle, rri, rri_dev = DB.calc_RSA(self.output['time'].astype('float'),
                                                              self.output['hr'][:,0].astype('int32'),
                                                              self.output['tv'], calc_var=False)
            self.output['rri_times'] = rri_times
            self.output['rri'] = rri
            self.output['rri_dev'] = rri_dev
            self.output['rri_ave'] = rri.mean()
            self.output['resp_cycle'] = resp_cycle
            if output_key in ['rsa', 'cvc']:
                self.output[output_key].update({ 'rri_dev': rri_dev,
                                                 'resp_cycle': resp_cycle,
                                                 'rri_ave': self.output['rri_ave'] })
        if output_key in ['ave_resp', 'cvc']:
            insp_times, exp_times, mean_insp, mean_exp = DB.mean_insp_exp(self.output['time'], self.output['tv'])
            self.output[output_key].update({'exp_time': mean_exp,
                                            'insp_time': mean_insp,
                                            'exp_times': exp_times,
                                            'insp_times': insp_times})
        if output_key == 'cvc':
            beat_latency = DB.calc_CVC(self.output['time'].astype('float'),
                                    self.output['hr'][:,0].astype('int32'),
                                    self.output['tv'])
            cdf = np.linspace(0, 1, beat_latency.size)
            self.output['cvc'].update({ 'beat_latency': beat_latency,
                                        'cdf': cdf })
        return self.output[output_key]

    @staticmethod
    def f(v):
        vlow, vhigh = -50, -20
        if v < vlow: return 0
        elif v > vhigh: return 1
        else: return (v-vlow)/(vhigh-vlow)
    
    @staticmethod
    def _calcExp(time, posti, tv):
        # find times when expiration begins (post i)
        p = np.zeros((tv.size, 2))
        kp, tpre = 0, 0
        thresh = 0.1
        for i in range(1, tv.size):
            if ModelOutput.f(posti[i-1]) >= thresh and ModelOutput.f(posti[i]) < thresh:
                p[kp,:] = [i, 1.0 / (time[i] - tpre)]
                kp += 1
                tpre = time[i]
        p = p[1:kp-1,1]
        return p
    
    @staticmethod
    def _calcInsp(time, earlyi, tv):
        # find times when inspiration begins (early i)
        p = np.zeros((tv.size, 2))
        kp, tpre = 0, 0
        for i in range(1, tv.size):
            if ModelOutput.f(earlyi[i-1]) >= 0.1 and ModelOutput.f(earlyi[i]) < 0.1:
                p[kp,:] = [i, 1.0 / (time[i] - tpre)]
                kp += 1
                tpre = time[i]
        p = p[1:kp-1,1]
        return p

    @staticmethod
    def _calcHR(time, hr_rate):
        thresh = 0.0001
        def beat(phi): return np.power(((1+np.sin(phi*2.0*np.pi))/2.0), 2.0)
        hr = np.zeros((time.size,2))
        k, tpre = 0, 0
        for i in range(1, hr_rate.size):
            if beat(hr_rate[i-1]) < thresh and beat(hr_rate[i]) >= thresh:
                hr[k,1] = 1.0 / (time[i] - tpre)
                hr[k,0] = i
                k += 1
                tpre=time[i]
        hr = hr[1:k-1,:]
        return hr

    def _setupOutput(self):
        self.output['preii'] = self.array[:,0]
        self.output['earlyi'] = self.array[:,1]
        self.output['auge'] = self.array[:,2]
        self.output['posti'] = self.array[:,3]
        self.output['rampi'] = self.array[:,4]

        self.output['preii_nap'] = self.array[:,5]
        self.output['earlyi_iad'] = self.array[:,6]
        self.output['auge_iad'] = self.array[:,7]
        self.output['posti_iad'] = self.array[:,8]

        self.output['time'] = self.time
        self.output['hr_rate'] = self.array[:,10]
        
        self.output['hr'] = ModelOutput._calcHR(self.output['time'], self.output['hr_rate'])
        self.output['hb'] = self.output['hr'][:,0].astype('int32') # heart beat indices
        
        self.output['bp'] = self.array[:,11]
        
        self.output['bp_s'] = self.array[:,12]

        tv = self.array[:,9]
        
        # print(tv)
        
        other_sigs, time, tv_s = DB.smooth_signal(self.time, tv,
                                                  other_sigs=[self.output['bp'],
                                                              self.output['bp_s'],
                                                              self.output['hr']],
                                                  wavelet_size=10)

        self.output['time'] = time
        
        bp_conv = 120.0 / other_sigs[0].max()
        self.output['bp'] = other_sigs[0]*bp_conv

        bp_convs = 120.0 / other_sigs[1].max()
        self.output['bp_s'] = other_sigs[1]*bp_convs

        self.output['hr'] = other_sigs[2]

        tv_convs = 2.0 / tv_s.max()
        self.output['tv'] = tv_s*tv_convs

    def save_model(self, fn):
        np.save(fn, self.output)
        print("saved model definition to : {}".format(fn))
