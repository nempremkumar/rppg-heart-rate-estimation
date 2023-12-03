import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

multispectral_file_path = '/Users/premkumargudipudi/Documents/main_project/sample_main/20211008_101816_multispec.h5'


class HeartRateExtractor:

    def __init__(self, multispectral_frames):
        self.multispectral_frames = multispectral_frames

    @staticmethod
    def read_multispectral_data(file_path):
        with h5py.File(file_path, 'r') as f:
            multispectral_frames = np.array(f['Frames'])
        return multispectral_frames

    @staticmethod
    def extract_bandwidths(frames):
        bandwidth_videos = []
        for i in range(4):
            for j in range(4):
                bandwidth_video = frames[:, i::4, j::4]
                bandwidth_videos.append(bandwidth_video)
        return bandwidth_videos

    def get_roi(self, frame):
        h, w = frame.shape
        start_row, end_row = int(h * 0.25), int(h * 0.75)
        start_col, end_col = int(w * 0.25), int(w * 0.75)
        return frame[start_row:end_row, start_col:end_col]

    def temporal_signal_extraction(self, bandwidth_video):
        return [np.mean(self.get_roi(frame)) for frame in bandwidth_video]

    @staticmethod
    def smooth_signal(signal, window_size=3):
        return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

    @staticmethod
    def bandpass_filter(signal, lowcut=0.8, highcut=4.0, fs=30, order=1):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def method_fft_multispectral(self, bandwidth_video):
        temporal_signal = self.temporal_signal_extraction(bandwidth_video)
        smoothed_signal = self.smooth_signal(temporal_signal)
        filtered_signal = self.bandpass_filter(smoothed_signal)
        freqs, fft_values = self.get_fft(filtered_signal)
        return freqs, fft_values

    @staticmethod
    def get_fft(signal, fs=30):
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        fft_values = np.fft.rfft(signal)
        return freqs, fft_values
    @staticmethod
    def get_heart_rate_from_fft(freqs, fft_values, fs=30):
        dominant_frequency = freqs[np.argmax(np.abs(fft_values))]
        return dominant_frequency * 60 
    

    def plot_multispectral_data(self, bandwidth_videos, title_prefix, output_file):
        wavelengths = [804.69, 819.79, 834.79, 851.35, 738.52, 756.39, 772.20, 787.99,
                    668.11, 685.81, 702.82, 719.90, 804.72, 612.31, 627.51, 649.41]
        fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True) 
        frame_rate = 30  
        
        for idx, ax in enumerate(axs.ravel()):
            temporal_signal = self.temporal_signal_extraction(bandwidth_videos[idx])
            normalized_signal = (temporal_signal - np.min(temporal_signal)) / (np.max(temporal_signal) - np.min(temporal_signal))
            time = np.arange(len(temporal_signal)) / frame_rate
            ax.plot(time, normalized_signal, label=f"Band {idx+1}-{wavelengths[idx]}nm")
            ax.legend(loc='upper right')
        
        # Set common labels
        fig.text(0.5, 0.02, 'Time (s)', ha='center', va='center')
        fig.text(0.02, 0.5, 'Normalized Intensity', ha='center', va='center', rotation='vertical')
        plt.tight_layout(rect=[0.05, 0.05, 1, 1])
        plt.show()

    def plot_fft_data(self, bandwidth_videos, title_prefix, output_file):
        fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)  
        
        for idx, ax in enumerate(axs.ravel()):
            freqs, fft_values = self.method_fft_multispectral(bandwidth_videos[idx])
            heart_rate = self.get_heart_rate_from_fft(freqs, fft_values)
            ax.plot(freqs, np.abs(fft_values), label=f"Band {idx+1}-{heart_rate:.2f} BPM")
            ax.legend(loc='upper right')
            ax.set_xlim(0, 4)
        
        # Set common labels
        fig.text(0.5, 0.02, 'Frequency (Hz)', ha='center', va='center')
        fig.text(0.02, 0.4, 'Magnitude', ha='center', va='center', rotation='vertical')
        plt.tight_layout(rect=[0.05, 0.05, 1, 1])
        plt.show()


# MULTISPECTRAL PROCESSING
multispectral_frames = HeartRateExtractor.read_multispectral_data(multispectral_file_path)
multispectral_extractor = HeartRateExtractor(multispectral_frames)
bandwidth_videos = multispectral_extractor.extract_bandwidths(multispectral_frames)

# Generate and save the temporal signal plots
multispectral_extractor.plot_multispectral_data(bandwidth_videos, "Multispectral Data", "Multispectral_Temporal")
plt.close() 

# Generate and save the FFT plots
multispectral_extractor.plot_fft_data(bandwidth_videos, "Multispectral Data", "Multispectral_FFT")
plt.close()  

