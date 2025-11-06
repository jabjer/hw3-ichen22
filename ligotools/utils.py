from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import windows
from numpy.typing import ArrayLike


# function to whiten data
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0, 2048, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

# function to keep the data within integer limits, and write to wavfile:
def write_wavfile(filename,fs,data):
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename,int(fs), d)

# function that shifts frequency of a band-passed signal
def reqshift(data,fshift=100,sample_rate=4096):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    y[0:nbins]=0.
    z = np.fft.irfft(y)
    return z

def plot_matched_filter_results(
    *,
    time: np.ndarray,
    timemax: float,
    SNR: np.ndarray,
    det: str,
    pcolor: str,
    eventname: str,
    plottype: str,
    tevent: float,
    strain_whitenbp: np.ndarray,
    template_match: np.ndarray,
    datafreq: np.ndarray,
    template_fft: np.ndarray,
    d_eff: float,
    freqs: np.ndarray,
    data_psd: np.ndarray,
    outdir: str = "figures",
    show: bool = True
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # SNR around event (wide + zoom)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time - timemax, SNR, pcolor, label=f"{det} SNR(t)")
    plt.grid(True)
    plt.ylabel("SNR")
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.legend(loc="upper left")
    plt.title(f"{det} matched filter SNR around event")

    plt.subplot(2, 1, 2)
    plt.plot(time - timemax, SNR, pcolor, label=f"{det} SNR(t)")
    plt.grid(True)
    plt.ylabel("SNR")
    plt.xlim([-0.15, 0.05])
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{eventname}_{det}_SNR.{plottype}")
    if show:
        plt.show()
    plt.close()

    # Whitened time series with template and residuals
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time - tevent, strain_whitenbp, pcolor, label=f"{det} whitened h(t)")
    plt.plot(time - tevent, template_match, "k", label="Template(t)")
    plt.ylim([-10, 10])
    plt.xlim([-0.15, 0.05])
    plt.grid(True)
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.ylabel("whitened strain (units of noise stdev)")
    plt.legend(loc="upper left")
    plt.title(f"{det} whitened data around event")

    plt.subplot(2, 1, 2)
    plt.plot(time - tevent, strain_whitenbp - template_match, pcolor, label=f"{det} resid")
    plt.ylim([-10, 10])
    plt.xlim([-0.15, 0.05])
    plt.grid(True)
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.ylabel("whitened strain (units of noise stdev)")
    plt.legend(loc="upper left")
    plt.title(f"{det} Residual whitened data after subtracting template around event")

    plt.tight_layout()
    plt.savefig(f"{outdir}/{eventname}_{det}_matchtime.{plottype}")
    if show:
        plt.show()
    plt.close()

    # ASD and template spectrum
    plt.figure(figsize=(10, 6))
    # must multiply by sqrt(f) to compare template fft on top of ASD
    template_f = np.abs(template_fft) * np.sqrt(np.abs(datafreq)) / max(d_eff, np.finfo(float).eps)
    plt.loglog(datafreq, template_f, "k", label="template(f)*sqrt(f)")
    plt.loglog(freqs, np.sqrt(data_psd), pcolor, label=f"{det} ASD")
    plt.xlim(20, np.max(datafreq))
    plt.ylim(1e-24, 1e-20)
    plt.grid(True, which="both")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("strain noise ASD (strain/rtHz), template h(f)*rt(f)")
    plt.legend(loc="upper left")
    plt.title(f"{det} ASD and template around event")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{eventname}_{det}_matchfreq.{plottype}")
    if show:
        plt.show()
    plt.close()