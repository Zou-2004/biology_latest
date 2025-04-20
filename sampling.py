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

def normalize_densities(densities, min_val, max_val):
    """
    Normalize density values to [0, 1] using provided min and max values.
    
    Args:
        densities: Array of density values.
        min_val: Minimum value for normalization.
        max_val: Maximum value for normalization.
    
    Returns:
        Normalized densities in range [0, 1].
    """
    densities = np.array(densities, dtype=np.float32)
    normalized = (densities - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero
    return normalized

def prepare_training_data(density_map):
    print(f"Original density map range: {density_map.min():.6f} to {density_map.max():.6f}")
    
    # Extract all significant values (above threshold in absolute terms)
    threshold = 0
    indices = np.where(np.abs(density_map) >= threshold)
    positions = np.stack([indices[0], indices[1], indices[2]], axis=1)
    
    # Get the original densities (unnormalized)
    densities = density_map[indices]
    orig_min, orig_max = densities.min(), densities.max()
    
    # Normalize densities to [0, 1]
    normalized_densities = normalize_densities(densities, orig_min, orig_max)
    
    # Store normalization parameters for potential denormalization
    norm_params = {
        'original_min': orig_min,
        'original_max': orig_max,
        'normalized_min': normalized_densities.min(),
        'normalized_max': normalized_densities.max(),
    }
    
    print("\nData statistics:")
    print(f"Number of points: {len(positions)}")
    print(f"Original density range: {orig_min:.6f} to {orig_max:.6f}")
    print(f"Normalized density range: {normalized_densities.min():.6f} to {normalized_densities.max():.6f}")
    
    return positions, normalized_densities, norm_params

def save_combined_file(input_dir, output_path):
    input_dir = Path(input_dir)
    
    # Search for all common cryo-EM file formats
    patterns = ['*.mrc', '*.map', '*.mrc.gz', '*.map.gz', '*.ccp4']
    map_files = []
    for pattern in patterns:
        map_files.extend(input_dir.glob(pattern))
    
    # Remove duplicates
    map_files = list(set(map_files))
    
    if not map_files:
        raise FileNotFoundError(f"No map files found in {input_dir} with extensions {patterns}")
    
    print(f"Found {len(map_files)} map files: {[f.name for f in map_files]}")
    
    with h5py.File(output_path, 'w') as f:
        for map_path in map_files:
            print(f"\nProcessing {map_path}")
            # Handle file key for .gz or regular extensions
            if map_path.suffix == '.gz':
                file_key = map_path.stem  # Removes .gz
                file_key = Path(file_key).stem  # Removes .mrc or .map
            else:
                file_key = map_path.stem  # Removes .mrc, .map, or .ccp4
            
            # Read the map file
            try:
                data = mrcfile.read(map_path).copy()
            except Exception as e:
                print(f"Error reading {map_path}: {str(e)}")
                continue
            
            # Process data with normalization
            positions, densities, norm_params = prepare_training_data(data)
            
            if len(positions) > 0:
                group = f.create_group(file_key)
                group.attrs['filename'] = file_key
                group.attrs['original_shape'] = data.shape
                
                # Save normalization parameters
                for key, value in norm_params.items():
                    if value is not None:
                        group.attrs[key] = value
                
                normalized_pos = normalize_coordinates(positions, data.shape)
                group.create_dataset('points', data=normalized_pos, compression='gzip')
                group.create_dataset('values', data=densities[:, np.newaxis], compression='gzip')
                group.attrs['num_points'] = len(positions)

if __name__ == "__main__":
    input_dir = "/home/zcy/seperate_VAE/data"
    output_path = "/home/zcy/seperate_VAE/density_data.hdf5"
    save_combined_file(input_dir, output_path)
