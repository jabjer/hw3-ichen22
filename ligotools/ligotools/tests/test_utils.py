import numpy as np
import os
from scipy.io import wavfile
from ligotools import utils

def test_whiten():
    x = np.zeros(4096)
    dt = 1/4096
    interp_psd = lambda f: np.ones_like(f)
    y = utils.whiten(x, interp_psd, dt)
    assert y.shape == x.shape
    assert np.allclose(y, 0.0)

def test_write_wavfile(tmp_path):
    fs = 4096
    x = np.linspace(-1, 1, fs)
    filename = tmp_path / "test.wav"

    utils.write_wavfile(filename, fs, x)

    assert filename.exists()
    assert os.path.getsize(filename) > 0

    rate, data = wavfile.read(filename)
    assert rate == fs
    assert np.all(np.isfinite(data))

def test_reqshift():
    fs = 4096
    T = 1.0
    t = np.arange(0, T, 1/fs)
    f0 = 100.0
    fshift = 150.0
    x = np.sin(2*np.pi*f0*t)

    z = utils.reqshift(x, fshift=fshift, sample_rate=fs)

    X = np.fft.rfft(x)
    Z = np.fft.rfft(z)
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    f_x = freqs[np.argmax(np.abs(X))]
    f_z = freqs[np.argmax(np.abs(Z))]

    assert np.isclose(f_x, f0, atol=2.0)       
    assert np.isclose(f_z, f0 + fshift, atol=5.0) 