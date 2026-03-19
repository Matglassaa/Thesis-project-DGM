import os
import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

def groupFacies(fac, grouping_scheme=None):
    """
    Groups facies produced by the flumy simulation.
    Default scheme maps into FA1 (1), FA2 (7), and FA3 (8).
    """
    fac_int = fac.astype(np.int32)
    lut = np.zeros(256, dtype=np.uint8)

    if grouping_scheme is None:
        lut[0:4] = 1       # FA1: Fluvial and distributary channel deposits -> fac 0,1,2 & 3
        lut[4:8] = 7       # FA2: Crevasse splays and levee deposits -> fac 4,5,6 & 7
        lut[8:13] = 8      # FA3: Overbank deposists and paleosols -> fac 8,9 -> rest does not occur often 
    else:
        for new_val, old_vals in grouping_scheme.items():
            lut[list(old_vals)] = new_val

    return lut[fac_int]

def save_as_npz(data_dict, filepath):
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    np.savez_compressed(filepath, **data_dict)
    return filepath

def save_as_h5(data_dict, filepath):
    if not H5PY_AVAILABLE:
        raise ImportError("h5py library required to save as .h5.")
    if not filepath.endswith(('.h5', '.hdf5')):
        filepath += '.h5'
    with h5py.File(filepath, 'w') as hf:
        for key, value in data_dict.items():
            hf.create_dataset(key, data=value)
    return filepath

def save_sample(data_dict, output_dir, sample_name, save_format='npz'):
    os.makedirs(output_dir, exist_ok=True)
    base_filepath = os.path.join(output_dir, sample_name)

    if save_format == 'npz':
        return save_as_npz(data_dict, base_filepath)
    elif save_format in ['h5', 'hdf5']:
        return save_as_h5(data_dict, base_filepath)
    else:
        raise ValueError("Unsupported save format. Use 'npz' or 'h5'.")