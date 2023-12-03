"""
This project incorporates contributions from various sources:
- A significant portion of the signal extraction algorithms were developed with assistance from Instructor and ChatGPT by OpenAI.
- Subsequent modifications, refinements, and application-specific adjustments were carried out by me to better suit the project's requirements and objectives.

It's essential to recognize that while ChatGPT provided foundational guidance, the final implementation and results were achieved through a collaborative effort, blending external guidance with personal expertise,instructor and project-specific knowledge.
"""

import h5py
import cv2
import numpy as np
import os

class DataProcessor:
    def __init__(self, thermal_file_path, multispectral_file_path):
        self.thermal_file_path = thermal_file_path
        self.multispectral_file_path = multispectral_file_path
        self.thermal_frames, self.thermal_timestamps = self.read_thermal_data(self.thermal_file_path)
        self.nir_frames, self.nir_timestamps = self.read_multispectral_data(self.multispectral_file_path)
        self.bandwidth_videos = self.extract_bandwidths(self.nir_frames)
        self.thermal_sync_indices, self.nir_sync_indices = self.get_synchronized_indices()

    def read_thermal_data(self, file_path):
        with h5py.File(file_path, 'r') as f:
            thermal_frames = np.array(f['Frames'])
            thermal_timestamps = np.array(f['Timestamps_ms'])
        return thermal_frames, thermal_timestamps

    def read_multispectral_data(self, file_path):
        with h5py.File(file_path, 'r') as f:
            nir_frames = np.array(f['Frames'])
            nir_timestamps = np.array(f['Timestamps_ms'])
        return nir_frames, nir_timestamps

    def extract_bandwidths(self, frames):
        bandwidth_videos = []
        for i in range(4):
            for j in range(4):
                bandwidth_video = frames[:, i::4, j::4]
                bandwidth_videos.append(bandwidth_video)
        return bandwidth_videos

    def get_thermal_display_frame(self, frame):
        normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colormapped = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        return colormapped
    
    def get_synchronized_indices(self, threshold=20):
        thermal_sync_indices = []
        nir_sync_indices = []

        t_idx = 0
        n_idx = 0
        while t_idx < len(self.thermal_timestamps) and n_idx < len(self.nir_timestamps):
            if abs(self.thermal_timestamps[t_idx] - self.nir_timestamps[n_idx]) <= threshold:
                thermal_sync_indices.append(t_idx)
                nir_sync_indices.append(n_idx)
                t_idx += 1
                n_idx += 1
            elif self.thermal_timestamps[t_idx] < self.nir_timestamps[n_idx]:
                t_idx += 1
            else:
                n_idx += 1

        return thermal_sync_indices, nir_sync_indices
    

    def display_combined_video(self, output_filename='output_video.avi'):
        # Determine the dimensions of the combined display
        sample_frame = self.get_thermal_display_frame(self.thermal_frames[0])
        sample_frame_resized = cv2.resize(sample_frame, (sample_frame.shape[1] * 3 // 2, sample_frame.shape[0]))
        height = sample_frame_resized.shape[0] + self.bandwidth_videos[0][0].shape[0] * 4
        width = sample_frame_resized.shape[1] + self.bandwidth_videos[0][0].shape[1] * 4

        # Initialize video writer using 'XVID' codec for AVI format
        height, width = 2048, 3648
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, 20.0, (width, height))
        total_frames_written = 0

        try:
            for t_idx, n_idx in zip(self.thermal_sync_indices, self.nir_sync_indices):
                thermal_display = self.get_thermal_display_frame(self.thermal_frames[t_idx])
                thermal_display = cv2.normalize(thermal_display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                grid_display = []
                for i in range(4):
                    row_frames = [self.bandwidth_videos[i*4 + j][n_idx] for j in range(4)]
                    row_frames_rotated = [cv2.transpose(cv2.flip(frame, flipCode=0)) for frame in row_frames]
                    row_frames_normalized = [cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for frame in row_frames_rotated]
                    row_display = np.hstack(row_frames_normalized)
                    grid_display.append(row_display)

                multispectral_display = np.vstack(grid_display)
                multispectral_display = cv2.cvtColor(multispectral_display, cv2.COLOR_GRAY2BGR)
                thermal_display = cv2.resize(thermal_display, (thermal_display.shape[1] * multispectral_display.shape[0] // thermal_display.shape[0], multispectral_display.shape[0]))
                combined_display = np.hstack((thermal_display, multispectral_display))

                # Debugging print statements
                print(f"thermal_display dimensions: {thermal_display.shape}")
                print(f"multispectral_display dimensions: {multispectral_display.shape}")
                print(f"combined_display dimensions: {combined_display.shape}")
                print(f"Expected dimensions: height={height}, width={width}")

                
                # Write the frame to the video writer
                out.write(combined_display)
                total_frames_written += 1

                cv2.imshow('Thermal & Multispectral Display', combined_display)

                if cv2.waitKey(60) & 0xFF == ord('q'):
                    break
        finally:
            out.release()
            cv2.destroyAllWindows()


    def access_subject_data(self, base_directory, subject_id):
        # Construct the paths dynamically based on the subject_id
        thermal_path = f"{base_directory}/{subject_id}/{subject_id}_FLIRAX5.h5"
        multispectral_path = f"{base_directory}/{subject_id}/{subject_id}_multispec.h5"
        
        self.thermal_frames, self.thermal_timestamps = self.read_thermal_data(thermal_path)
        self.nir_frames, self.nir_timestamps = self.read_multispectral_data(multispectral_path)


thermal_file_path = '/Users/premkumargudipudi/Documents/main_project/sample_main/20211008_101816_FLIRAX5.h5'
multispectral_file_path = '/Users/premkumargudipudi/Documents/main_project/sample_main/20211008_101816_multispec.h5'
output_h5_file = '/Users/premkumargudipudi/Documents/3/plots.h5'  
processor = DataProcessor(thermal_file_path, multispectral_file_path)
processor.display_combined_video("output_video.avi")