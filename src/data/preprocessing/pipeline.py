import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data.preprocessing.base.modify_data import ModifyData
from src.data.preprocessing.extractor.extractor import Extractor
from src.utils.paths import get_project_path

class Pipeline:
    def __init__(self, num_cycle, inhibitor_name, split, norm_feat=False):
        
        self.split = split
        self.modify_data = ModifyData(num_cycle, inhibitor_name, split)

        self.vol_orig = self.modify_data.vol.copy()
        self.cur_orig = self.modify_data.cur.copy()
        self.ppm = self.modify_data.ppm.to_list()
        self.inhibitor = self.modify_data.inhibitor.to_list()
        self.number_of_cycle = self.modify_data.number_of_cycle.to_list() 
        
        rdkit = pd.read_csv(os.path.join(get_project_path(), "data", "mol_rdkit.csv")).drop(columns=["SMILES"])
        dft = pd.read_csv(os.path.join(get_project_path(), "data", "mol_dft.csv"))
        
        if norm_feat == True:
            
            desc_scaler = MinMaxScaler()
            ppm_scaler = MinMaxScaler()

            self.desc = pd.merge(rdkit, dft, how="left", on=["Inhibitor"])
            
            norm_dd = pd.DataFrame(desc_scaler.fit_transform(self.desc.drop(columns=["Inhibitor"])))
            norm_dd.columns = self.desc.drop(columns=["Inhibitor"]).columns
            norm_dd.columns = self.desc.drop(columns=["Inhibitor"]).columns
            norm_dd.insert(0, 'Inhibitor', self.desc['Inhibitor'])
            
            ppm_scaled = ppm_scaler.fit_transform(np.array(self.ppm).reshape(-1, 1)).flatten()
            
            # ppm_scaled = pd.Series(np.log(self.ppm)).replace(-np.inf, 0)

            f_1 = pd.concat([
                self.cur_orig,
                pd.DataFrame({"Inhibitor": self.inhibitor}),
                pd.DataFrame({"ppm": ppm_scaled})
            ], axis=1)

            self.full_data = f_1.merge(norm_dd, how="left", on="Inhibitor")
        
        else:
            self.desc = pd.merge(rdkit, dft, how="left", on=["Inhibitor"])
        
            f_1 = pd.concat([self.cur_orig, pd.DataFrame({"Inhibitor":self.inhibitor}), pd.DataFrame({"ppm":self.ppm})], axis=1)
            self.full_data = f_1.merge(self.desc, how="left", on="Inhibitor")
    
        self._step_1_get_data()
        self._step_2_modify_data()
        self._step_3_extract_data() 

    def _step_1_get_data(self):
        self.vol = self.modify_data.vol
        self.cur = self.modify_data.cur
    
    def _step_2_modify_data(self):
        self.vol_modify = self.modify_data.vol_modify
        self.cur_modify = self.modify_data.cur_modify
    
    def _step_3_extract_data(self):
        self.analyzed_data, self.vol_final, self.cur_final = Extractor(
            self.vol_modify, 
            self.cur_modify
        ).get_data()

        self.analyzed_data["ppm"] = self.modify_data.ppm
        self.analyzed_data["Inhibitor"] = self.modify_data.inhibitor

    