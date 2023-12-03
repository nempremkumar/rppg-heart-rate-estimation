import h5py
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import interp1d

class HeartRateExtractorFromThermal:

    def __init__(self, thermal_frames):
        self.thermal_frames = thermal_frames

    @staticmethod
    def read_thermal_data(file_path):
        with h5py.File(file_path, 'r') as f:
            thermal_frames = np.array(f['Frames'])
        return thermal_frames

    def get_roi(self, frame):
        normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        face_cascade_path = '/Users/premkumargudipudi/Documents/main_project/haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        faces = face_cascade.detectMultiScale(normalized_frame, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi = frame[y:y+h, x:x+w]
            # Normalize the ROI
            roi_normalized = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
            return roi_normalized
        else:
            # If no face is detected, normalize and return the whole frame
            return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    def temporal_signal_extraction(self):
        # Extract the temporal signal from the normalized ROIs
        return [np.mean(self.get_roi(frame)) for frame in self.thermal_frames]

    def bandpass_filter(self, signal, lowcut=0.67, highcut=4.0, fs=50, order=1):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def get_heart_rate(self, signal, fs=50):
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        fft_values = np.fft.rfft(signal)
        dominant_frequency = freqs[np.argmax(np.abs(fft_values))]
        return dominant_frequency * 60

    def save_combined_plot(self, temporal_signal, filtered_signal, heart_rate):
        max_val = np.max(temporal_signal)
        min_val = np.min(temporal_signal)
        normalized_temporal_signal = (temporal_signal - min_val) / (max_val - min_val)
        # Convert time intervals to seconds for plotting
        frame_rate = 50  
        time_intervals_in_seconds = [i / frame_rate for i in range(len(temporal_signal))]
        interpolation_function = interp1d(time_intervals_in_seconds, normalized_temporal_signal, kind='cubic', fill_value="extrapolate")

        # Interpolate on a denser time scale
        dense_time_intervals = np.linspace(time_intervals_in_seconds[0], time_intervals_in_seconds[-1], num=len(time_intervals_in_seconds) * 10)
        dense_normalized_signal = interpolation_function(dense_time_intervals)

        
        # Plot the normalized temporal signal
        plt.figure(figsize=(10, 6))
        plt.plot(time_intervals_in_seconds, normalized_temporal_signal, 'o', label='Original Data', markersize=4) 
        plt.plot(dense_time_intervals, dense_normalized_signal, label='Interpolated Temporal Signal')
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Intensity")
        #plt.title("Normalized Temporal Signal")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot the FFT results
        plt.figure(figsize=(10, 6))
        freqs = np.fft.rfftfreq(len(filtered_signal), 1/frame_rate)
        fft_values = np.fft.rfft(filtered_signal)
        plt.plot(freqs, np.abs(fft_values), color='orange', label='FFT')
        plt.axvline(x=heart_rate/60, color='red', linestyle='--', label=f"Heart Rate: {heart_rate:.0f} BPM")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        #plt.title("FFT Analysis")
        plt.legend()
        plt.tight_layout()
        plt.show()
       

    def process(self):
        temporal_signal = self.temporal_signal_extraction()
        filtered_signal = self.bandpass_filter(temporal_signal)
        heart_rate = self.get_heart_rate(filtered_signal)
        return temporal_signal, filtered_signal, heart_rate

# calling
thermal_file_path = '/Users/premkumargudipudi/Documents/main_project/sample_main/20211008_101816_FLIRAX5.h5'
thermal_frames = HeartRateExtractorFromThermal.read_thermal_data(thermal_file_path)
extractor = HeartRateExtractorFromThermal(thermal_frames)
temporal_signal, filtered_signal, heart_rate = extractor.process()
print(f"Estimated Heart Rate using FFT: {heart_rate:.2f} BPM")
extractor.save_combined_plot(temporal_signal, filtered_signal, heart_rate)