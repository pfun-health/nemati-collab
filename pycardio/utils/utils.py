import numpy as np
from scipy.optimize import curve_fit

def calc_stderr(vv, nn):
    """ vv is variance, nn is number of data """
    return np.sqrt(vv) / np.sqrt(nn)

def calc_rot(rri_ave, exp_time, insp_time):
    return int( np.floor( (1.0 / rri_ave) / ( 1.0 / (exp_time+insp_time) ) ) )

def simple_fourier(phase, A, B):
    """ first-order fourier estimate for RSA """
    return A*np.cos(2*np.pi*phase) + B*np.sin(2*np.pi*phase)

def complex_fourier(phase, *params):
    result = np.array([simple_fourier(phase, params[i-1], B) for i, B in enumerate(params[1:])], dtype=float)
    result = result.sum(0)
    return result

def interp_cvc(vec_data, cdf_data, vec_model, cdf_model):
    
    model_interp = [cdf_model[0]]
    for ik, xk in enumerate(vec_data[1:-1]):
        xl = vec_data[vec_data < xk][-1]
        xr = vec_data[vec_data >= xk][0]
        yl = cdf_model[np.argmin(np.abs(xl - vec_model))]
        yr = cdf_model[np.argmin(np.abs(xr - vec_model))]
        cdf = yl + (yr - yl) * (xk - xl) / (xr - xl)
        model_interp.append(cdf)
    model_interp.append(cdf_model[-1])

    model_interp = np.array(model_interp, dtype=float).reshape(-1)

    data_interp = cdf_data
    phi = vec_data
        
    return phi, data_interp, model_interp

def calc_pdf(x):
    sigma = calc_stderr(x, x.size)
    return np.exp(-np.power((x - x.mean())/2*sigma, 2)) / (sigma*np.pi*2)
