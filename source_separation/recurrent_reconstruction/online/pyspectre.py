from __future__ import division
import numpy as np
import scipy


def spectre(audio, win=512, poverlap=.75):
    overlap = np.floor(win * poverlap)
    nfft = win
    l = len(audio)
    w = scipy.hanning(win+2)[1:-1]
    position = 0
    count = 0
    spec = np.zeros((nfft, np.floor((l - win) / (win-overlap) + 1)))
    phase = np.zeros_like(spec)
    while position + win - 1 <= l:
        y = audio[position:position+win] * w
        tmp_fft = np.fft.fft(y, nfft)
        spec[:, count] = np.abs(tmp_fft)
        phase[:, count] = np.angle(tmp_fft)
        position += win - overlap
        count += 1
    spec = spec[: np.ceil((nfft + 1) / 2), :]
    return spec, phase


def inv_spectre(spec, phase, poverlap=.75):
    win = (spec.shape[0] - 1) * 2
    nfft = win
    overlap = np.floor(win * poverlap)
    a = spec[::-1]
    spec = np.concatenate((spec, a[1:-1, :]))
    n = 0
    w = scipy.hanning(win+2)[1:-1]
    signal = np.zeros_like(spec)
    while n < spec.shape[1]:
        signal[:, n] = np.real(np.fft.ifft(np.exp(1j*phase[:, n]) * (spec[:, n]), nfft)) * w
        n += 1

    f_signal = np.zeros((spec.shape[1]-1)*(win-overlap) + win)
    normalization = np.zeros_like(f_signal)
    step = win - overlap
    for k in range(spec.shape[1]):
        f_signal[k * step: win + k * step] = f_signal[k * step: win + k * step] + signal[:, k]
        normalization[k * step: win + k * step] = normalization[k * step: win + k * step] + w
    signal = f_signal / (overlap / win * normalization)
    return signal


def snr_audio(original, recon):
    m = min(len(original), len(recon))
    original = original[:m]
    recon = recon[:m]
    return 10 * np.log10((np.sum(original**2) / np.sum((original - recon)**2)))
