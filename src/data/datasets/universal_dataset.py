import torch
from torch.utils.data import Dataset

descriptors_name = ['MolWt', 'MolLogP', 'NumRotatableBonds', 'TPSA',
       'FractionCSP3', 'NumAromaticRings', 'Chi0', 'Chi1', 'Kappa1', 'Kappa2',
       'BertzCT', 'BalabanJ', 'PEOE_VSA1', 'PEOE_VSA2', 'SMR_VSA1', 'SMR_VSA2',
       'EState_VSA1', 'EState_VSA2', 'MaxEStateIndex', 'MinEStateIndex',
       'FpDensityMorgan1', 'SMR_VSA10', 'VSA_EState2', 'VSA_EState3',
       'BCUT2D_MWHI', 'BCUT2D_LOGP', 'BCUT2D_CHGHI', 'Nitrogen', 'Sulfur',
       'AmineGroup', 'G Eh', 'HOMO eV', 'LUMO eV', 'μ D',
       'Final entropy term Eh', 'Total enthalpy Eh', 'Electronic entropy Eh',
       'Vibrational entropy Eh', 'Rotational entropy Eh',
       'Translational entropy Eh', 'ppm']

class CVADataset(Dataset):
    def __init__(self, current):
        if "Inhibitor" in current.columns:
            current = current.drop(columns=["Inhibitor"])
            
        self.vah = current.drop(columns=current.columns[968:]).astype("float32").values
        self.desc = current[descriptors_name].astype("float32").values

        self.vahh = torch.tensor(self.vah, dtype=torch.float32)
        self.descc = torch.tensor(self.desc, dtype=torch.float32)

    def __len__(self):
        return len(self.vahh)

    def __getitem__(self, idx):
        return {
            "vah": self.vahh[idx],
            "features": self.descc[idx]
        }