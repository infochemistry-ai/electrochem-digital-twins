import os
import numpy as np
import pandas as pd
import pywt
from src.data.preprocessing.base.base_data import BaseData

class ModifyData(BaseData):
    def __init__(self, num_cycle, inhibitor_name, split):
        super().__init__(num_cycle, inhibitor_name, split)

        self.split = split
        self.num_cycle = num_cycle
        self.inhibitor_name = inhibitor_name

        self.vol_modify = self.vol.copy()
        self.cur_modify = self.cur.copy()
        
        self._remove_tails()
        self._wavelet_interpolation()

    def _remove_tails(self):

        def _fill_voltage(voltage):
            voltage_new = voltage.copy()
            volt_increase = 0.008333999

            for index, element in voltage_new.iterrows():
                for i in range(element.shape[0]):
                    if pd.isna(element[i]):
                        voltage_new.iloc[index, i] = voltage_new.iloc[index, i-1] + volt_increase * np.sign(voltage_new.iloc[index, i-1])

            return voltage_new

        indices_to_nan = [(150, 333), (634, 815), (939, 967)]
        
        voltage_new, current_new = [], []

        for index, volt_new in _fill_voltage(self.vol_modify).iterrows():

            curr_new = self.cur_modify.iloc[index].copy()

            for start, end in indices_to_nan:
                curr_new.iloc[start:end] = np.nan

            curr_new = curr_new.interpolate()

            voltage_new.append(volt_new)
            current_new.append(curr_new)

        voltage_new = pd.DataFrame(voltage_new)
        current_new = pd.DataFrame(current_new)

        voltage_new.columns = range(voltage_new.shape[1])
        current_new.columns = range(current_new.shape[1])

        self.vol_modify, self.cur_modify = voltage_new, current_new

    def _wavelet_interpolation(self):
        wavelet='db4' 
        level=2

        vol_int, cur_int = [], []

        for i in range(len(self.vol_modify)):
            coeffs_vol = pywt.wavedec(self.vol_modify.iloc[i], wavelet, level=level)
            vol_resampled = pywt.upcoef('a', coeffs_vol[0], wavelet, level=level, take=968)
            
            coeffs_cur = pywt.wavedec(self.cur_modify.iloc[i], wavelet, level=level)
            cur_resampled = pywt.upcoef('a', coeffs_cur[0], wavelet, level=level, take=968)

            vol_int.append(vol_resampled)
            cur_int.append(cur_resampled)

        
        
        self.vol_modify, self.cur_modify = pd.DataFrame(vol_int), pd.DataFrame(cur_int)