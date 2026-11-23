import numpy as np
from ctypes import c_int
import gc
from ..utils import calc_stderr, calc_rot, interp_cvc
# import matplotlib.pyplot as plt

def CF_simple(f_data, f_model, var_data=None, N_data=None):
    """ Simple cost function that returns the cost,
    which is computed as a comparison between the
    data and the model output.

    Input arguments:
    f_data (numpy.array or float) : the estimate from the data
      (e.g. the average heart rate)
    f_model (numpy.array or float) : the estimate from the model
      (e.g. the model's average heart rate)
    var_data [optional] (float) : the variance estimated from the
      data, which can be computed from the array f_data if it is
      provided.
    """
    
    if isinstance(f_data, np.ndarray) and f_data.ndim != 1:
        raise RuntimeError("f_data should be 1 dimensional.")
    if isinstance(f_model, np.ndarray) and f_model.ndim != 1:
        raise RuntimeError("f_model should be 1 dimensional.")
    
    if isinstance(var_data, np.ndarray): variance = var_data
    elif var_data is not None: variance = var_data
    
    elif all([not isinstance(var_data, np.ndarray), isinstance(f_data, np.ndarray)]):
        variance = np.power(calc_stderr(f_data.var(),f_data.size), 2)
        
    elif all([not isinstance(var_data, np.ndarray), not isinstance(f_data, np.ndarray)]):
        if not N_data and not var_data:
            raise RuntimeError("You must either give the variance estimate (var_data),\
            or f_data must be a numpy array, which is used to calculate the variance.")
        else:
            variance = var_data
            
    if isinstance(f_model, np.ndarray) and f_model.size > f_data.size:
        f_model = f_model[:f_data.size]

    N = 1.0
    if N_data: N = N_data # scale the cost function by the number of data
    
    cost = np.sum(np.power((f_data - f_model), 2) / variance) * N

    if np.isnan(cost): cost = np.inf # Set nan costs to inf

    return cost

def CF_rsa(rsa_data, rsa_model, var_data=None):
    
    model_resp = rsa_model['resp_cycle']
    model_rri = rsa_model['rri_dev']

    data_resp = rsa_data['resp_cycle']
    data_rri = rsa_data['rri_dev']

    var_data = rsa_data['rsa_var']

    try:
        if not data_resp.size or not model_resp.size:
            return np.inf
        model_order = np.argsort(model_resp)
        model_rri = np.interp(data_resp, model_resp[model_order],
                              model_rri[model_order])

        cost1 = CF_simple(data_rri, model_rri, var_data=var_data) # RRI_dev cost
        
        var_rri = var_data # Variance of the RRI for the selected data
        cost2 = CF_simple(rsa_data['rri'].mean(), rsa_model['rri_ave'],
                          var_data=var_rri) # Average RRI cost
        
        # print(data_rri.size, cost1, cost2)

        return cost1 + cost2

    except (ValueError, ZeroDivisionError) as excep:
        print(excep)
        return np.inf

def CF_resp(resp_data, resp_model, var_data=None):

    insp_cost = CF_simple(resp_data['insp_time'], resp_model['insp_time'],
                          var_data=resp_data['insp_var'],
                          N_data=len(resp_data['insp_times']))

    exp_cost = CF_simple(resp_data['exp_time'], resp_model['exp_time'],
                         var_data=resp_data['exp_var'],
                         N_data=len(resp_data['exp_times']))

    return (insp_cost + exp_cost)*10.0

def CF_cvc(cvc_data, cvc_model, var_data=None):
    """
    cdf is defined as K/N, where K is the index of the sorted array
    and N is the number of datapoints.
    Then n*D^2 is the cost function, where D is the max distance.
    """

    cdf_data = cvc_data['cdf']
    vec_data = cvc_data['beat_latency']
    
    cdf_model = cvc_model['cdf']
    vec_model = cvc_model['beat_latency']

    if cdf_model.size <= 1 or\
       any(rt == 0.0 for rt in [cvc_model['exp_time'],
                                cvc_model['insp_time'],
                                cvc_model['rri_ave']]):
        return np.inf


    cdf_cost = 0
    rot_cost = 0
    N = cdf_model.size
    
    try:
        data_rot = calc_rot(cvc_data['rri'].mean(),
                            cvc_data['exp_time'],
                            cvc_data['insp_time'])
        model_rot = calc_rot(cvc_model['rri_ave'],
                             cvc_model['exp_time'],
                             cvc_model['insp_time'])
        rot_cost = np.abs(data_rot - model_rot) * np.sqrt(N)
    except ValueError as excep:
        return np.inf

    try:
        interp_vec, cdf_data, cdf_model = interp_cvc(vec_data, cdf_data, vec_model, cdf_model)
    except IndexError:
        return np.inf
    
    N = interp_vec.size
    D = np.max(np.abs(cdf_data - cdf_model))
    Dmax = np.max(np.abs(cdf_data - interp_vec))
    cdf_cost = np.sqrt(N) * (1 + (np.power(D + 1, 2) / np.power(Dmax + 1, 2)))

    rsa_cost = CF_rsa(cvc_data, cvc_model)
    resp_cost = CF_resp(cvc_data, cvc_model)

    cost = cdf_cost + rot_cost + rsa_cost + resp_cost

    # print("cvc: {:.3f} ; rsa: {:.3f} ; resp: {:.3f}".format(cdf_cost, rsa_cost, resp_cost))

    if np.isnan(cost): cost = np.inf

    return cost
