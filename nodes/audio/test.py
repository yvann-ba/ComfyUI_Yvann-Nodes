import numpy as np


audio_peaks_index = np.array([1, 2, 3, 4, 5], dtype=int)
audio_peaks_index = np.insert(audio_peaks_index, 0, 0)
str_peaks_index = ', '.join(map(str, audio_peaks_index))
print(str_peaks_index)