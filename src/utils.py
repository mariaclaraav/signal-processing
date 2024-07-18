from numpy import sqrt, pi, log10
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, welch


# Function to convert FFT to power levels in dB
def db20(W, Nfft=None):
    'Given FFT, return power level in dB'
    if Nfft is None:  # Assume W is FFT
        return 20 * log10(abs(W))
    else:  # Assume time-domain passed, so need FFT
        DFT = fft(W, Nfft) / sqrt(Nfft)
        return 20 * log10(abs(DFT))
    
# Butterworth filter design function
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Filter the data with the Butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_psd(signal, fs, order, ax, cutoff):
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    ax.semilogy(f, Pxx, label=f'Order {order}')
    plt.axvline(cutoff, color='k', linestyle='--', linewidth=1)

def plot_fft(time, signal, fs, title='FFT of Signal'):
    N = len(time)
    
    # Calculate FFT
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)[:N//2]
    
    # Remove negative part and DC component
    yf = yf[1:N//2]
    xf = xf[1:]
    
    # Plot
    plt.figure(figsize=(10, 3))
    plt.plot(xf, 2.0/N * np.abs(yf))
    plt.grid()
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()
    
# Function to calculate and plot the PSD
def plot_psd(signal, sampling_rate, windows, nperseg, noverlap, axis_name):
    plt.figure(figsize=(12, 4))
    
    for window_type in windows:
        # Calculate the PSD using the welch function
        f, Pxx = welch(signal, fs=sampling_rate, window=window_type, nperseg=nperseg, noverlap=noverlap)
        
        # Plot the PSD
        plt.semilogy(f, Pxx, label=f'Window: {window_type}')
    
    plt.title(f'PSD - {axis_name}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'Power Spectral Density $[V^2/Hz]$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
