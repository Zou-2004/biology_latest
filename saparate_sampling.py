import mrcfile
import numpy as np
import h5py
from pathlib import Path

def normalize_coordinates(positions, shape):
    positions = np.array(positions, dtype=np.float32)
    normalized = positions / np.array(shape)
    print(f"Coordinate ranges before normalization: {positions.min(axis=0)} to {positions.max(axis=0)}")
    print(f"Shape used for normalization: {shape}")
    print(f"Coordinate ranges after normalization: {normalized.min(axis=0)} to {normalized.max(axis=0)}")
    return normalized

def normalize_density(values, mode='positive'):
    """Normalize density values to [0,1] range and store normalization parameters"""
    min_val = values.min()
    max_val = values.max()
    normalized = (values - min_val) / (max_val - min_val)
    return normalized, {'min': float(min_val), 'max': float(max_val), 'mode': mode}

def prepare_separate_training_data(density_map, pos_threshold, neg_threshold):
    print(f"Original density map range: {density_map.min():.6f} to {density_map.max():.6f}")
    
    # Create separate maps for positive and negative values
    pos_map = density_map.copy()
    pos_map[pos_map <= pos_threshold] = 0
    
    neg_map = density_map.copy()
    neg_map[neg_map >= -neg_threshold] = 0
    neg_map = np.abs(neg_map)  # Convert to positive values for training
    
    # Extract positive coordinates and values
    pos_indices = np.where(pos_map > pos_threshold)
    pos_positions = np.stack([pos_indices[0], pos_indices[1], pos_indices[2]], axis=1)
    pos_densities = pos_map[pos_indices]
    
    # Extract negative coordinates and values (stored as positive)
    neg_indices = np.where(neg_map > neg_threshold)
    neg_positions = np.stack([neg_indices[0], neg_indices[1], neg_indices[2]], axis=1)
    neg_densities = neg_map[neg_indices]
    
    # Normalize densities
    norm_pos_densities, pos_norm_params = normalize_density(pos_densities, 'positive')
    norm_neg_densities, neg_norm_params = normalize_density(neg_densities, 'negative')
    
    print("\nPositive data statistics:")
    print(f"Number of positive points: {len(pos_positions)}")
    print(f"Positive density range before normalization: {pos_densities.min():.6f} to {pos_densities.max():.6f}")
    print(f"Positive density range after normalization: {norm_pos_densities.min():.6f} to {norm_pos_densities.max():.6f}")
    
    print("\nNegative data statistics:")
    print(f"Number of negative points: {len(neg_positions)}")
    print(f"Negative density range before normalization: {neg_densities.min():.6f} to {neg_densities.max():.6f}")
    print(f"Negative density range after normalization: {norm_neg_densities.min():.6f} to {norm_neg_densities.max():.6f}")
    
    return (pos_positions, norm_pos_densities, pos_norm_params), (neg_positions, norm_neg_densities, neg_norm_params)

def save_separate_files(input_dir, pos_output_path, neg_output_path, pos_threshold, neg_threshold):
    input_dir = Path(input_dir)
    mrc_files = list(input_dir.glob('*.mrc'))
    
    # Track global normalization parameters
    global_pos_params = {'min': float('inf'), 'max': float('-inf')}
    global_neg_params = {'min': float('inf'), 'max': float('-inf')}
    
    # Create separate HDF5 files for positive and negative data
    with h5py.File(pos_output_path, 'w') as pos_f, h5py.File(neg_output_path, 'w') as neg_f:
        for mrc_path in mrc_files:
            print(f"\nProcessing {mrc_path}")
            file_key = mrc_path.stem
            data = mrcfile.read(mrc_path)
            
            # Split and process data
            (pos_positions, norm_pos_densities, pos_params), \
            (neg_positions, norm_neg_densities, neg_params) = \
                prepare_separate_training_data(data, pos_threshold, neg_threshold)
            
            # Update global parameters
            global_pos_params['min'] = min(global_pos_params['min'], pos_params['min'])
            global_pos_params['max'] = max(global_pos_params['max'], pos_params['max'])
            global_neg_params['min'] = min(global_neg_params['min'], neg_params['min'])
            global_neg_params['max'] = max(global_neg_params['max'], neg_params['max'])
            
            # Save positive data
            if len(pos_positions) > 0:
                pos_group = pos_f.create_group(file_key)
                pos_group.attrs['filename'] = file_key
                pos_group.attrs['original_shape'] = data.shape
                pos_group.attrs.update(pos_params)  # Save normalization parameters
                
                normalized_pos = normalize_coordinates(pos_positions, data.shape)
                pos_group.create_dataset('points', data=normalized_pos, compression='gzip')
                pos_group.create_dataset('values', data=norm_pos_densities[:, np.newaxis], compression='gzip')
                pos_group.attrs['num_points'] = len(pos_positions)
            
            # Save negative data
            if len(neg_positions) > 0:
                neg_group = neg_f.create_group(file_key)
                neg_group.attrs['filename'] = file_key
                neg_group.attrs['original_shape'] = data.shape
                neg_group.attrs.update(neg_params)  # Save normalization parameters
                
                normalized_neg = normalize_coordinates(neg_positions, data.shape)
                neg_group.create_dataset('points', data=normalized_neg, compression='gzip')
                neg_group.create_dataset('values', data=norm_neg_densities[:, np.newaxis], compression='gzip')
                neg_group.attrs['num_points'] = len(neg_positions)
        
        # Save global normalization parameters
        pos_f.attrs.update(global_pos_params)
        neg_f.attrs.update(global_neg_params)

def verify_separate_files(pos_path, neg_path):
    # Verify positive data
    print("\nVerifying positive data file:")
    with h5py.File(pos_path, 'r') as f:
        file_keys = list(f.keys())
        print(f"Total positive files: {len(file_keys)}")
        print(f"Global positive density range: {f.attrs['min']:.6f} to {f.attrs['max']:.6f}")
        
        for key in file_keys:
            points = f[key]['points'][:]
            values = f[key]['values'][:]
            
            print(f"\nFile: {key}")
            print(f"Points shape: {points.shape}")
            print(f"Values shape: {values.shape}")
            print(f"Points range: {points.min(axis=0)} to {points.max(axis=0)}")
            print(f"Values range: {values.min():.6f} to {values.max():.6f}")
            print(f"Original density range: {f[key].attrs['min']:.6f} to {f[key].attrs['max']:.6f}")
    
    # Verify negative data
    print("\nVerifying negative data file:")
    with h5py.File(neg_path, 'r') as f:
        file_keys = list(f.keys())
        print(f"Total negative files: {len(file_keys)}")
        print(f"Global negative density range: {f.attrs['min']:.6f} to {f.attrs['max']:.6f}")
        
        for key in file_keys:
            points = f[key]['points'][:]
            values = f[key]['values'][:]
            
            print(f"\nFile: {key}")
            print(f"Points shape: {points.shape}")
            print(f"Values shape: {values.shape}")
            print(f"Points range: {points.min(axis=0)} to {points.max(axis=0)}")
            print(f"Values range: {values.min():.6f} to {values.max():.6f}")
            print(f"Original density range: {f[key].attrs['min']:.6f} to {f[key].attrs['max']:.6f}")

if __name__ == "__main__":
    input_dir = "/home/zcy/random_sampling/data"
    pos_output_path = "/home/zcy/seperate_VAE/positive_density_data.hdf5"
    neg_output_path = "/home/zcy/seperate_VAE/negative_density_data.hdf5"
    pos_threshold = 0
    neg_threshold = 0
    save_separate_files(input_dir, pos_output_path, neg_output_path, pos_threshold, neg_threshold)
    verify_separate_files(pos_output_path, neg_output_path)