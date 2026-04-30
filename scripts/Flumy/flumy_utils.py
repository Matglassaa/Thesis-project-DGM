import os
import numpy as np

def groupFacies(fac, grouping_scheme=None):
    """
    Groups facies produced by the flumy simulation.
    Default scheme maps into FA1 (1), FA2 (7), and FA3 (8).
    """
    fac_int = fac.astype(np.int32)
    lut = np.zeros(256, dtype=np.uint8)

    if grouping_scheme is None:
        lut[0:4] = 1       # FA1: Fluvial and distributary channel deposits -> fac 0,1,2 & 3
        lut[4:8] = 2       # FA2: Crevasse splays and levee deposits -> fac 4,5,6 & 7
        lut[8:13] = 3      # FA3: Overbank deposists and paleosols -> fac 8,9 -> rest does not occur often 
    else:
        for new_val, old_vals in grouping_scheme.items():
            lut[list(old_vals)] = new_val

    return lut[fac_int]

def save_sample(data_dict, output_dir, sample_name, save_format='npz'):
    """
    Saves the dictionary of arrays to the specified format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if save_format == 'npz':
        # Saves one file: sample_X.npz containing 'facies' and 'age' arrays
        filepath = os.path.join(output_dir, f"{sample_name}.npz")
        np.savez_compressed(filepath, **data_dict)
        return filepath
        
    elif save_format == 'npy':
        # Saves multiple files: sample_X_facies.npy and sample_X_age.npy
        saved_paths = []
        for key, value in data_dict.items():
            filepath = os.path.join(output_dir, f"{sample_name}_{key}.npy")
            np.save(filepath, value)
            saved_paths.append(filepath)
        return saved_paths
        
    else:
        raise ValueError("Unsupported save format. Use 'npz' or 'npy'.")