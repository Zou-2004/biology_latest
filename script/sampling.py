import mrcfile
import numpy as np
import h5py
from pathlib import Path
import os
import torch

def normalize_coordinates(positions, shape, use_gpu=False):
    if use_gpu:
        positions = torch.tensor(positions, dtype=torch.float32, device='cuda')
        shape_tensor = torch.tensor(shape, dtype=torch.float32, device='cuda')
        normalized = positions / shape_tensor
        print(f"Coordinate ranges before normalization: {positions.min(dim=0)[0].cpu()} to {positions.max(dim=0)[0].cpu()}")
        print(f"Shape used for normalization: {shape}")
        print(f"Coordinate ranges after normalization: {normalized.min(dim=0)[0].cpu()} to {normalized.max(dim=0)[0].cpu()}")
        return normalized.cpu().numpy()
    else:
        positions = np.array(positions, dtype=np.float32)
        normalized = positions / np.array(shape)
        print(f"Coordinate ranges before normalization: {positions.min(axis=0)} to {positions.max(axis=0)}")
        print(f"Shape used for normalization: {shape}")
        print(f"Coordinate ranges after normalization: {normalized.min(axis=0)} to {normalized.max(axis=0)}")
        return normalized

def prepare_training_data(density_map, threshold, use_gpu=False):
    print(f"Original density map range: {density_map.min():.6f} to {density_map.max():.6f}")
    
    if use_gpu:
        density_map = torch.tensor(density_map, device='cuda')
        indices = torch.where(density_map > threshold)
        positions = torch.stack([indices[0], indices[1], indices[2]], dim=1)
        densities = density_map[indices].cpu().numpy()
        binary_map = torch.zeros(density_map.shape, dtype=torch.uint8, device='cuda')
        binary_map[indices] = 1
        binary_map = binary_map.cpu().numpy()
        total_elements = density_map.numel()  # Use numel() for PyTorch tensors
    else:
        indices = np.where(density_map > threshold)
        positions = np.stack([indices[0], indices[1], indices[2]], axis=1)
        densities = density_map[indices]
        binary_map = np.zeros(density_map.shape, dtype=np.uint8)
        binary_map[indices] = 1
        total_elements = density_map.size  # Use size for NumPy arrays
    
    print("\nData statistics:")
    print(f"Number of points with density above {threshold}: {len(positions)}")
    print(f"Density range: {densities.min():.6f} to {densities.max():.6f}")
    print(f"Percentage of points retained: {100 * len(positions) / total_elements:.2f}%")
    print(f"Binary map shape: {binary_map.shape}, Non-zero count: {np.sum(binary_map)}, dtype: {binary_map.dtype}")
    print(f"Binary map size in memory: {binary_map.nbytes:,} bytes ({binary_map.nbytes / 1024**2:.2f} MB)")
    
    return positions, densities, binary_map

def save_combined_file(input_dir, output_path, threshold, use_gpu=False):
    input_dir = Path(input_dir)
    mrc_files = list(input_dir.glob('*.mrc'))
    
    if not mrc_files:
        raise FileNotFoundError(f"No MRC files found in {input_dir}")
    
    print(f"Found {len(mrc_files)} MRC files: {[f.name for f in mrc_files]}")
    
    with h5py.File(output_path, 'w') as f:
        for idx, mrc_path in enumerate(mrc_files):
            print(f"\nProcessing {mrc_path}, progress {idx+1}/{len(mrc_files)}")
            file_key = mrc_path.stem
            try:
                data = mrcfile.read(mrc_path)
            except Exception as e:
                print(f"Error reading {mrc_path}: {str(e)}")
                continue
            
            positions, densities, binary_map = prepare_training_data(data, threshold, use_gpu)
            
            if len(positions) > 0:
                group = f.create_group(file_key)
                group.attrs['filename'] = str(mrc_path.name)
                group.attrs['original_shape'] = data.shape
                
                normalized_pos = normalize_coordinates(positions, data.shape, use_gpu)
                
                group.create_dataset('points', data=normalized_pos, compression='gzip', compression_opts=9)
                group.create_dataset('values', data=densities[:, np.newaxis], compression='gzip', compression_opts=9)
                group.create_dataset('binary_map', data=binary_map, dtype='uint8', compression='gzip', compression_opts=9)
                group.attrs['num_points'] = len(positions)
                
                print(f"Saved {len(positions)} positive density points and binary map for {file_key}")
            else:
                print(f"Warning: No positive density points found in {file_key}")

def verify_file(output_path):
    with h5py.File(output_path, 'r') as f:
        file_keys = list(f.keys())
        print(f"\nVerifying file structure for {output_path}")
        print(f"Number of file groups: {len(file_keys)}")
        if len(file_keys) > 0:
            first_key = file_keys[0]
            print(f"First file key: {first_key}")
            if 'points' in f[first_key]:
                print(f"✓ 'points' dataset exists with shape {f[first_key]['points'].shape}")
            if 'values' in f[first_key]:
                print(f"✓ 'values' dataset exists with shape {f[first_key]['values'].shape}")
            if 'binary_map' in f[first_key]:
                binary_shape = f[first_key]['binary_map'].shape
                binary_dtype = f[first_key]['binary_map'].dtype
                non_zeros = np.sum(f[first_key]['binary_map'][:])
                print(f"✓ 'binary_map' dataset exists with shape {binary_shape}, dtype {binary_dtype}, {non_zeros:,} non-zero points")

def compress_binary_maps(hdf5_path, output_dir):
    import gzip
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    with h5py.File(hdf5_path, 'r') as f:
        for file_key in f.keys():
            binary_map = f[file_key]['binary_map'][:]
            print(f"Before compression - dtype: {binary_map.dtype}, size: {binary_map.nbytes:,} bytes ({binary_map.nbytes / 1024**2:.2f} MB)")
            
            output_file = output_dir / f"{file_key}_binary_map.gz"
            with gzip.open(output_file, 'wb', compresslevel=9) as gz_file:
                gz_file.write(binary_map.tobytes())
            
            uncompressed_size = binary_map.nbytes
            compressed_size = output_file.stat().st_size
            print(f"Compressed binary map for {file_key} to {output_file}")
            print(f"Uncompressed size: {uncompressed_size:,} bytes ({uncompressed_size / 1024**2:.2f} MB)")
            print(f"Compressed size: {compressed_size:,} bytes ({compressed_size / 1024**2:.2f} MB)")
            print(f"Compression ratio: {uncompressed_size / compressed_size:.2f}x")

if __name__ == "__main__":
    input_dir = "/home/zcy/NFD/denoised_sample/"  # Fixed typo in path
    output_path = "/home/zcy/seperate_VAE/density_data.hdf5"
    compressed_dir = "/home/zcy/seperate_VAE/compressed_binary_maps"
    os.makedirs(compressed_dir, exist_ok=True)
    
    # import shutil
    # if Path(compressed_dir).exists():
    #     shutil.rmtree(compressed_dir)
    os.makedirs(compressed_dir,exist_ok=True)
    
    threshold = 0.04
    use_gpu = torch.cuda.is_available()

    print(f"Using GPU: {use_gpu}")
    
    # save_combined_file(input_dir, output_path, threshold, use_gpu)
    # verify_file(output_path)
    compress_binary_maps(output_path, compressed_dir)
    
    print("\nProcessing complete!")
    print(f"Dataset saved to: {output_path}")
    print(f"Compressed binary maps saved to: {compressed_dir}")