import os
import pywt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.paths import get_project_path

class BaseData:
    def __init__(self, num_cycle, inhibitor_name, split):

        self.split = split
        self.num_cycle = num_cycle
        self.inhibitor_name = inhibitor_name

        self.voltage_ar, self.current_ar, self.prop = self.get_data()
        self.vol_inter, self.cur_inter = self.wavelet_interpolation(self.voltage_ar, self.current_ar)

        self.cur = pd.concat([pd.DataFrame(self.cur_inter), self.prop], axis=1)
        self.vol = pd.concat([pd.DataFrame(self.vol_inter), self.prop], axis=1)

        self._select_cycle()
        self._select_inhibitor()

        self.number_of_cycle = self.cur['Num_of_Cycle']
        self.ppm = self.cur['ppm']
        self.inhibitor = self.cur['Inhibitor']
        self.cur = self.cur.drop(columns=['ppm', 'Num_of_Cycle', 'Inhibitor']) * 1000
        self.vol = self.vol.drop(columns=['ppm', 'Num_of_Cycle', 'Inhibitor'])
        
    def get_data(self):
        list_files = []
        for root, dirs, files in os.walk(os.path.join(get_project_path(), 'data', 'experimental_data')):
            for file in files:
                if file.endswith('.edf'):
                    list_files.append(os.path.join(root, file))

        if self.split == "all":
            list_files = list_files
        elif self.split == "train":
            list_files, _ = train_test_split(list_files, test_size=0.1, random_state=2683)
        elif self.split == "test":
            _, list_files = train_test_split(list_files, test_size=0.1, random_state=2683)


        current, voltage, conc, inh = [], [], [], []

        num_of_cycle_df = []

        for file in list_files:
            with open(file, encoding='latin-1') as f:
                data = f.readlines()
                
            current_temp, volt_temp = [], []

            num_of_cycle = 0

            for line in data:
                values = line.strip().split()

                if len(values) == 4:
                    current_temp.append(float(values[3]))
                    volt_temp.append(float(values[2]))

                elif values[0] == 'de':
                    num_of_cycle_df.append(num_of_cycle)
                    num_of_cycle += 1
                    
                    current.append(current_temp)
                    voltage.append(volt_temp)

                    conc.append(int(file.strip().split('/')[-2].split()[0]))
                    inh.append(file.strip().split('/')[-3].split()[0])

                    current_temp, volt_temp = [], []

                else:
                    continue


        drop_idx = []
        for i in range(len(current)):
            if len(current[i]) < 900:
                drop_idx.append(i)


        current = [cur for i, cur in enumerate(current) if i not in drop_idx]
        voltage = [val for i, val in enumerate(voltage) if i not in drop_idx]
        
        conc = [ppm for i, ppm in enumerate(conc) if i not in drop_idx]
        inh = [inhib for i, inhib in enumerate(inh) if i not in drop_idx]
        num_of_cycle_df = [cyc for i, cyc in enumerate(num_of_cycle_df) if i not in drop_idx]

        prop = pd.DataFrame(
            {
                "ppm": conc, 
                "Inhibitor": inh, 
                "Num_of_Cycle": num_of_cycle_df
            }
        )

        return voltage, current, prop
    

    def wavelet_interpolation(self, vol, cur, wavelet='db4', level=2):

        vol_int, cur_int = [], []
        
        for i in range(len(vol)):
            coeffs_vol = pywt.wavedec(vol[i], wavelet, level=level)
            vol_resampled = pywt.upcoef('a', coeffs_vol[0], wavelet, level=level, take=968)
            
            coeffs_cur = pywt.wavedec(cur[i], wavelet, level=level)
            cur_resampled = pywt.upcoef('a', coeffs_cur[0], wavelet, level=level, take=968)
            
            vol_int.append(vol_resampled)
            cur_int.append(cur_resampled)
        
        return np.array(vol_int), np.array(cur_int)
    
    def _select_cycle(self):
        self.cur = self.cur[self.cur['Num_of_Cycle'].isin(self.num_cycle)].reset_index(drop=True)
        self.vol = self.vol[self.vol['Num_of_Cycle'].isin(self.num_cycle)].reset_index(drop=True)

    def _select_inhibitor(self):
        if self.inhibitor_name == 'all':
            self.cur = self.cur.reset_index(drop=True)
            self.vol = self.vol.reset_index(drop=True)
        else:
            self.cur = self.cur[self.cur['Inhibitor'] == self.inhibitor_name].reset_index(drop=True)
            self.vol = self.vol[self.vol['Inhibitor'] == self.inhibitor_name].reset_index(drop=True)
