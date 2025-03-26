import mrcfile
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter, zoom

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
    """
    densities = np.array(densities, dtype=np.float32)
    normalized = (densities - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon
    return normalized

def prepare_training_data(density_map, resolution_label="high_res"):
    print(f"{resolution_label} density map range: {density_map.min():.6f} to {density_map.max():.6f}")
    
    # Extract all significant values (above threshold in absolute terms)
    threshold = 0
    indices = np.where(np.abs(density_map) >= threshold)
    positions = np.stack([indices[0], indices[1], indices[2]], axis=1)
    
    # Get the original densities (unnormalized)
    densities = density_map[indices]
    orig_min, orig_max = densities.min(), densities.max()
    
    # Normalize densities to [0, 1]
    normalized_densities = normalize_densities(densities, orig_min, orig_max)
    
    # Store normalization parameters
    norm_params = {
        'original_min': orig_min,
        'original_max': orig_max,
        'normalized_min': normalized_densities.min(),
        'normalized_max': normalized_densities.max(),
    }
    
    print(f"\n{resolution_label} Data statistics:")
    print(f"Number of points: {len(positions)}")
    print(f"Original density range: {orig_min:.6f} to {orig_max:.6f}")
    print(f"Normalized density range: {normalized_densities.min():.6f} to {normalized_densities.max():.6f}")
    
    return positions, normalized_densities, norm_params

def downsample_mrc(data, target_res_factor=10):
    """
    Downsample density map to simulate lower resolution (e.g., 1Å to 10Å).
    target_res_factor: Factor to reduce resolution (e.g., 10 for 1Å → 10Å).
    """
    # Smooth to avoid aliasing
    sigma = target_res_factor / 2  # Adjust sigma based on downsampling factor
    smoothed = gaussian_filter(data, sigma=sigma)
    
    # Downsample
    zoom_factor = 1 / target_res_factor
    downsampled = zoom(smoothed, zoom_factor, order=1)  # Linear interpolation
    
    print(f"Downsampled shape: {downsampled.shape}")
    return downsampled

def save_combined_file(input_dir, output_path, low_res_factor=10):
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
            # Handle file key
            if map_path.suffix == '.gz':
                file_key = Path(map_path.stem).stem  # Removes .gz and .mrc/.map
            else:
                file_key = map_path.stem  # Removes .mrc, .map, or .ccp4
            
            # Read the high-resolution map
            try:
                high_res_data = mrcfile.read(map_path).copy()
            except Exception as e:
                print(f"Error reading {map_path}: {str(e)}")
                continue
            
            # Generate low-resolution data
            low_res_data = downsample_mrc(high_res_data, target_res_factor=low_res_factor)
            
            # Process high-resolution data
            high_pos, high_dens, high_norm_params = prepare_training_data(high_res_data, "high_res")
            
            # Process low-resolution data
            low_pos, low_dens, low_norm_params = prepare_training_data(low_res_data, "low_res")
            
            if len(high_pos) > 0 and len(low_pos) > 0:
                # Create group for this file
                group = f.create_group(file_key)
                group.attrs['filename'] = file_key
                group.attrs['high_res_shape'] = high_res_data.shape
                group.attrs['low_res_shape'] = low_res_data.shape
                
                # High-resolution subgroup
                high_group = group.create_group('high_res')
                for key, value in high_norm_params.items():
                    if value is not None:
                        high_group.attrs[key] = value
                high_normalized_pos = normalize_coordinates(high_pos, high_res_data.shape)
                high_group.create_dataset('points', data=high_normalized_pos, compression='gzip')
                high_group.create_dataset('values', data=high_dens[:, np.newaxis], compression='gzip')
                high_group.attrs['num_points'] = len(high_pos)
                
                # Low-resolution subgroup
                low_group = group.create_group('low_res')
                for key, value in low_norm_params.items():
                    if value is not None:
                        low_group.attrs[key] = value
                low_normalized_pos = normalize_coordinates(low_pos, low_res_data.shape)
                low_group.create_dataset('points', data=low_normalized_pos, compression='gzip')
                low_group.create_dataset('values', data=low_dens[:, np.newaxis], compression='gzip')
                low_group.attrs['num_points'] = len(low_pos)

if __name__ == "__main__":
    input_dir = "/home/zcy/seperate_VAE/data"
    output_path = "/home/zcy/seperate_VAE/density_data.hdf5"
    save_combined_file(input_dir, output_path, low_res_factor=10)