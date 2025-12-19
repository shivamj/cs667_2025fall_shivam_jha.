"""
ECG signal preprocessing and noise injection utilities.
Implements Pan-Tompkins algorithm components and noise simulation.
"""

import numpy as np
from scipy import signal


def bandpass_filter(ecg_signal, fs=360, lowcut=5, highcut=15):
    """
    Bandpass filter for ECG signal preprocessing.
    
    Implements the filtering stage of the Pan-Tompkins algorithm.
    Suppresses baseline wander (< 5 Hz) and high-frequency noise (> 15 Hz).
    
    Args:
        ecg_signal: Raw ECG signal
        fs: Sampling frequency (Hz), default: 360 Hz (MIT-BIH standard)
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
    
    Returns:
        filtered_signal: Bandpass-filtered ECG signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # 4th order Butterworth bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    
    return filtered_signal


def derivative_filter(ecg_signal):
    """
    Five-point derivative filter for slope information.
    
    Part of Pan-Tompkins algorithm for QRS detection.
    y(nT) = (1/8T)[-x(nT-2T) - 2x(nT-T) + 2x(nT+T) + x(nT+2T)]
    
    Args:
        ecg_signal: Filtered ECG signal
    
    Returns:
        derivative: Derivative of the signal
    """
    # Five-point derivative approximation
    derivative = np.zeros_like(ecg_signal)
    
    for i in range(2, len(ecg_signal) - 2):
        derivative[i] = (-ecg_signal[i-2] - 2*ecg_signal[i-1] + 
                        2*ecg_signal[i+1] + ecg_signal[i+2]) / 8.0
    
    return derivative


def z_normalize(signal):
    """
    Z-normalization: zero mean, unit variance.
    
    Essential for DTW-based classification to ensure amplitude invariance.
    
    Args:
        signal: Input signal
    
    Returns:
        normalized: Z-normalized signal
    """
    mean = np.mean(signal)
    std = np.std(signal)
    
    if std < 1e-8:
        return signal - mean
    
    return (signal - mean) / std


def add_gaussian_noise(signal, snr_db):
    """
    Add Gaussian White Noise (GWN) at specified SNR level.
    
    SNR (dB) = 10 * log10(P_signal / P_noise)
    
    Args:
        signal: Clean signal
        snr_db: Desired Signal-to-Noise Ratio in decibels
    
    Returns:
        noisy_signal: Signal with added Gaussian white noise
    """
    # Calculate signal power
    signal_power = np.mean(signal ** 2)
    
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate required noise power
    noise_power = signal_power / snr_linear
    
    # Generate noise
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    
    return signal + noise


def calculate_snr(clean_signal, noisy_signal):
    """
    Calculate actual SNR between clean and noisy signals.
    
    Args:
        clean_signal: Original clean signal
        noisy_signal: Signal with noise
    
    Returns:
        snr_db: Signal-to-Noise Ratio in decibels
    """
    noise = noisy_signal - clean_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:
        return np.inf
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db


def segment_heartbeat(ecg_signal, r_peak_index, before=100, after=150):
    """
    Segment a single heartbeat centered on R-peak.
    
    Args:
        ecg_signal: Continuous ECG signal
        r_peak_index: Index of R-peak
        before: Number of samples before R-peak
        after: Number of samples after R-peak
    
    Returns:
        segment: Segmented heartbeat
    """
    start = max(0, r_peak_index - before)
    end = min(len(ecg_signal), r_peak_index + after)
    
    segment = ecg_signal[start:end]
    
    # Pad if necessary
    if len(segment) < (before + after):
        segment = np.pad(segment, (0, before + after - len(segment)), mode='edge')
    
    return segment


def preprocess_pipeline(ecg_signal, fs=360):
    """
    Complete preprocessing pipeline: Filter → Normalize.
    
    Args:
        ecg_signal: Raw ECG signal
        fs: Sampling frequency
    
    Returns:
        processed: Preprocessed ECG signal ready for classification
    """
    # Step 1: Bandpass filter
    filtered = bandpass_filter(ecg_signal, fs)
    
    # Step 2: Z-normalize
    normalized = z_normalize(filtered)
    
    return normalized


if __name__ == "__main__":
    # Test preprocessing pipeline
    np.random.seed(42)
    
    # Generate synthetic ECG
    fs = 360
    t = np.linspace(0, 2, 2 * fs)
    ecg = np.zeros_like(t)
    
    # Add QRS complexes
    for peak_time in [0.5, 1.0, 1.5]:
        ecg += 2.0 * np.exp(-100 * (t - peak_time)**2)
    
    # Add baseline wander
    ecg += 0.5 * np.sin(2 * np.pi * 0.5 * t)
    
    # Preprocess
    processed = preprocess_pipeline(ecg, fs)
    
    # Add noise
    noisy_20db = add_gaussian_noise(processed, 20)
    noisy_10db = add_gaussian_noise(processed, 10)
    
    # Verify SNR
    actual_snr_20 = calculate_snr(processed, noisy_20db)
    actual_snr_10 = calculate_snr(processed, noisy_10db)
    
    print("Preprocessing Pipeline Test")
    print("="*50)
    print(f"Original signal length:  {len(ecg)}")
    print(f"Processed signal range:  [{processed.min():.2f}, {processed.max():.2f}]")
    print(f"Processed mean:          {processed.mean():.2e}")
    print(f"Processed std:           {processed.std():.2f}")
    print(f"\nNoise injection:")
    print(f"  Target SNR: 20 dB → Actual: {actual_snr_20:.2f} dB")
    print(f"  Target SNR: 10 dB → Actual: {actual_snr_10:.2f} dB")
    print("="*50)
