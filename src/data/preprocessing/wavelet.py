import numpy as np
import pywt

def wavelet_interpolation(current, wavelet="db4", level=2, target_length=968):
    cur_int = []

    cur_array = np.atleast_2d(current)

    for i in range(cur_array.shape[0]):
        signal = cur_array[i, :]
        if len(signal) < pywt.Wavelet(wavelet).dec_len:
            raise ValueError(
                f"Signal length ({len(signal)}) is too short for the chosen wavelet"
            )
            
        coeffs_cur = pywt.wavedec(signal, wavelet, level=level)

        cur_resampled = pywt.upcoef(
            "a", coeffs_cur[0], wavelet, level=level, take=target_length
        )

        cur_int.append(cur_resampled)

    return np.array(cur_int)