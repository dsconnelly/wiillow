import os

import numpy as np
import scipy.signal as signal
import xarray as xr

def apply_butterworth(u):
    sos = signal.butter(9, 1 / 120, output='sos', fs=1)
    func = lambda a: signal.sosfilt(sos, a)
    
    return xr.apply_ufunc(
        func, u,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True
    ).transpose('time', 'pfull')

def load_qbo(casedir, n_years=24, insert=''):
    years = sorted([int(s) for s in os.listdir(casedir) if s.isdigit()])
    fnames = [f'{casedir}/{year:02}/atmos_{insert}daily.nc' for year in years]
    
    with xr.open_mfdataset(fnames, decode_times=False) as ds:
        u = ds['ucomp'].sel(pfull=slice(None, 100)).sel(lat=slice(-5, 5))
        if insert:
            u = u.groupby(u['time'].astype(int)).mean('time')

        u = u.isel(time=slice(-(n_years * 360), None))
        u = u.mean(('lat', 'lon'))
        
    return u.load()

def qbo_amplitude(u, level=20):
    u = u.sel(pfull=level, method='nearest').values
    
    return statistic_with_confidence(u, np.std)

def qbo_period(u, level=27):
    u = u.sel(pfull=level, method='nearest').values
    u_hat = np.fft.fft(u, n=int(2.5e6))
    
    freqs = np.fft.fftfreq(u_hat.shape[0])
    idx = (freqs > 0) & (freqs >= 1 / len(u))
    
    powers = abs(u_hat[idx])
    periods = (1 / freqs[idx]) / 30
    
    k = powers.argmax()
    period = periods[k]
    half_max = powers[k] / 2
    
    starts, = np.where((powers[:-1] < half_max) & (powers[1:] > half_max))
    ends, = np.where((powers[:-1] > half_max) & (powers[1:] < half_max))
    start = starts[starts < k][-1]
    end = ends[ends > k][0]
    
    left = periods[start]
    right = periods[end]
    width = max(left - period, period - right)
    
    return period, width

def qbo_statistics(u):
    u = apply_butterworth(u)
    period, period_err = qbo_period(u)
    amp, amp_err = qbo_amplitude(u)
    
    return period, period_err, amp, amp_err
    
def statistic_with_confidence(a, func, n_resamples=int(1e4), confidence=0.95):
    stats = np.zeros(n_resamples)
    for i in range(n_resamples):
        stats[i] = func(np.random.choice(a, size=a.shape[0]))
        
    center = func(a)
    m = int(((1 - confidence) / 2) * n_resamples)
    left, *_, right = abs(np.sort(stats)[m:(-m)] - center)
    width = max(left, right)
    
    return center, width
    