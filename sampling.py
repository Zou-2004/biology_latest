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

def prepare_training_data(density_map):
    print(f"Original density map range: {density_map.min():.6f} to {density_map.max():.6f}")
    
    # Extract all significant values (above threshold in absolute terms)
    threshold = 0
    indices = np.where(np.abs(density_map) >= threshold)
    positions = np.stack([indices[0], indices[1], indices[2]], axis=1)
    densities = density_map[indices]
    
    print("\nData statistics:")
    print(f"Number of points: {len(positions)}")
    print(f"Density range: {densities.min():.6f} to {densities.max():.6f}")
    
    return positions, densities

def save_combined_file(input_dir, output_path):
    input_dir = Path(input_dir)
    mrc_files = list(input_dir.glob('*.mrc'))
    
    with h5py.File(output_path, 'w') as f:
        for mrc_path in mrc_files:
            print(f"\nProcessing {mrc_path}")
            file_key = mrc_path.stem
            data = mrcfile.read(mrc_path)
            
            # Process data
            positions, densities = prepare_training_data(data)
            
            if len(positions) > 0:
                group = f.create_group(file_key)
                group.attrs['filename'] = file_key
                group.attrs['original_shape'] = data.shape
                
                normalized_pos = normalize_coordinates(positions, data.shape)
                group.create_dataset('points', data=normalized_pos, compression='gzip')
                group.create_dataset('values', data=densities[:, np.newaxis], compression='gzip')
                group.attrs['num_points'] = len(positions)

if __name__ == "__main__":
    input_dir = "/home/zcy/random_sampling/data"
    output_path="/home/zcy/seperate_VAE/density_data.hdf5"
    save_combined_file(input_dir, output_path,)
    # verify_separate_files(pos_output_path, neg_output_path)