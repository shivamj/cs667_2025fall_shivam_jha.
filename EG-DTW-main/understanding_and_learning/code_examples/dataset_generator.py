# Helper script to generate example datasets and utilities
import numpy as np
import matplotlib.pyplot as plt

def generate_simple_sequences():
    """Generate two simple sequences for DTW demonstration."""
    t = np.linspace(0, 2*np.pi, 50)
    seq1 = np.sin(t)
    seq2 = np.sin(t * 1.2 + 0.5)  # Slightly shifted and stretched
    return seq1, seq2

def generate_ecg_beat(beat_type='N', length=250):
    """
    Generate synthetic ECG beat.
    
    Args:
        beat_type: 'N' (normal), 'V' (PVC), 'L' (LBBB), 'R' (RBBB), 'A' (APC)
        length: Number of samples
    """
    t = np.linspace(0, 1, length)
    signal = np.zeros_like(t)
    
    if beat_type == 'N':  # Normal
        signal += 2.0 * np.exp(-200 * (t - 0.5)**2)  # QRS
        signal += 0.5 * np.exp(-100 * (t - 0.65)**2)  # T-wave
        signal += 0.3 * np.exp(-150 * (t - 0.35)**2)  # P-wave
    elif beat_type == 'V':  # PVC
        signal += 2.5 * np.exp(-80 * (t - 0.55)**2)  # Wide QRS
        signal -= 0.6 * np.exp(-100 * (t - 0.75)**2)  # Inverted T
    elif beat_type == 'L':  # LBBB
        signal += 1.8 * np.exp(-100 * (t - 0.5)**2)  # Wide QRS
        signal += 0.4 * np.exp(-80 * (t - 0.68)**2)  # T-wave
        signal += 0.3 * np.exp(-150 * (t - 0.35)**2)  # P-wave
    elif beat_type == 'R':  # RBBB
        signal += 1.5 * np.exp(-250 * (t - 0.48)**2)  # Split QRS
        signal += 1.5 * np.exp(-250 * (t - 0.53)**2)
        signal += 0.5 * np.exp(-100 * (t - 0.68)**2)  # T-wave
    elif beat_type == 'A':  # APC
        signal += 2.0 * np.exp(-200 * (t - 0.48)**2)  # QRS
        signal += 0.5 * np.exp(-100 * (t - 0.63)**2)  # T-wave
        signal += 0.4 * np.exp(-150 * (t - 0.30)**2)  # Early P
    
    # Add small noise
    signal += 0.05 * np.random.randn(len(t))
    return signal

def save_sample_dataset(filename='sample_ecg_data.npz'):
    """Create and save a small ECG dataset."""
    np.random.seed(42)
    
    dataset = {
        'signals': [],
        'labels': [],
        'class_names': []
    }
    
    beat_types = ['N', 'V', 'L', 'R', 'A']
    
    for beat_type in beat_types:
        for i in range(10):  # 10 samples per class
            signal = generate_ecg_beat(beat_type)
            dataset['signals'].append(signal)
            dataset['labels'].append(beat_types.index(beat_type))
            dataset['class_names'].append(beat_type)
    
    np.savez(
        filename,
        signals=np.array(dataset['signals']),
        labels=np.array(dataset['labels']),
        class_names=np.array(dataset['class_names'])
    )
    
    print(f"✅ Dataset saved to {filename}")
    print(f"   Samples: {len(dataset['signals'])}")
    print(f"   Classes: {len(beat_types)}")

if __name__ == "__main__":
    save_sample_dataset('../datasets/sample_ecg_data.npz')
