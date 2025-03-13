import mne
import numpy as np

# Load EEG data
raw = mne.io.read_raw_edf("data/S001R04.edf", preload=True)
raw.filter(8., 30., fir_design='firwin')  # Bandpass filter for motor imagery

# Extract epochs
events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, baseline=None, preload=True)

# Compute PSD (Power Spectral Density) as state features
psds = epochs.compute_psd(method="welch", fmin=8, fmax=30).get_data()

# Flatten PSD features for RL state input
X_psd = psds.reshape(psds.shape[0], -1)  # Shape: (num_epochs, num_features)

y_labels = np.random.randint(0, 2, size=(X_psd.shape[0],))  # Binary labels

#print(X_psd.shape)
#print(X_psd[0])
