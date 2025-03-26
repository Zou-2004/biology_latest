import torch
import numpy as np
import mrcfile
import os
from train import INR_Network  # Ensure this imports the updated INR_Network with Sigmoid
import h5py
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class DensityPredictor:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.point_dim = 3
        self.ef_dim = 128  # Match training config
        self.gf_dim = 128  # Match training config
        self.z_dim = 64    # Match training config
        self.chunk_size = 100000
        self.num_frequencies = 10

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize network
        self.network = INR_Network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim, self.num_frequencies).to(device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()

        # Load z-vectors and normalization parameters from the checkpoint
        self.z_vectors = checkpoint['z_vectors']  # Dictionary of z-vectors
        self.norm_params = checkpoint['norm_params']  # Dictionary of normalization parameters

        # Convert z-vectors to tensors on the correct device
        for fname in self.z_vectors:
            self.z_vectors[fname] = self.z_vectors[fname].to(device)

        if '00256_10.00Apx' in self.norm_params:
            print(f"Normalization parameters for 00256_10.00Apx: {self.norm_params['00256_10.00Apx']}")
        else:
            print("Warning: Normalization parameters for 00256_10.00Apx not found in checkpoint.")

    def predict_density_batched(self, coords, z_vector):
        try:
            total_points = coords.shape[0]
            predictions = np.zeros((total_points,), dtype=np.float32)
            
            if len(z_vector.shape) == 1:
                z_vector = z_vector.unsqueeze(0)  # Ensure z_vector is [1, z_dim]
            
            for start_idx in range(0, total_points, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_points)
                chunk_coords = coords[start_idx:end_idx]
                
                with torch.no_grad():
                    coords_tensor = torch.from_numpy(chunk_coords).float().to(self.device)
                    z_tensor = z_vector.to(self.device)
                    
                    # Get predictions (assumed [0, 1] with Sigmoid)
                    chunk_pred = self.network.generator(coords_tensor, z_tensor)
                    chunk_values = chunk_pred.cpu().numpy()
                    
                    # Store the predictions
                    predictions[start_idx:end_idx] = chunk_values.squeeze()
                
                del coords_tensor, chunk_pred
                torch.cuda.empty_cache()
                
            return predictions.reshape(-1, 1)
            
        except Exception as e:
            print(f"Error in predict_density_batched: {str(e)}")
            print(f"Input coords shape: {coords.shape}")
            print(f"Z vector shape: {z_vector.shape}")
            raise

def denormalize_densities(normalized_densities, norm_params):
    """Denormalize densities from [0, 1] range back to original range."""
    if not norm_params or 'original_min' not in norm_params or 'original_max' not in norm_params:
        print("Warning: Missing normalization parameters. Using normalized values as-is.")
        return normalized_densities
    
    orig_min = norm_params['original_min']
    orig_max = norm_params['original_max']
    
    # Assuming input is [0, 1], scale back to original range
    denormalized = normalized_densities * (orig_max - orig_min) + orig_min
    return denormalized

def create_density_map(checkpoint_path, output_dir, volume_shape):
    predictor = DensityPredictor(checkpoint_path=checkpoint_path)
    
    # Create integer coordinates
    x = np.arange(volume_shape[0])
    y = np.arange(volume_shape[1])
    z = np.arange(volume_shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Store integer coordinates for later use
    int_coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # Normalize coordinates for network input [0, 1]
    grid_points = int_coords / (np.array(volume_shape) - 1)
    
    print(f"Total grid points: {grid_points.shape[0]}")
    print(f"Volume shape: {volume_shape}")
    print(f"Normalized coordinate range: [{grid_points.min():.4f}, {grid_points.max():.4f}]")
    
    # Use z-vectors and normalization parameters from the checkpoint
    file_names = list(predictor.z_vectors.keys())
    print(f"Found {len(file_names)} files to process")
    
    for idx, file_name in tqdm(enumerate(file_names)):
        print(f"\nProcessing file: {file_name} ({idx+1}/{len(file_names)})")
        
        z_vector = predictor.z_vectors[file_name]
        print(f"Z vector shape: {z_vector.shape}")
        
        # Get file-specific normalization params from the checkpoint
        norm_params = predictor.norm_params.get(file_name, {})
        
        if not norm_params:
            print(f"Warning: No normalization parameters found for {file_name}. Using default parameters.")
            norm_params = {
                'original_min': -0.3,  # Adjust defaults based on your data
                'original_max': 0.2
            }
        
        print(f"Using normalization parameters: {norm_params}")
        
        try:
            # Initialize empty volume
            volume = np.zeros(volume_shape, dtype=np.float32)
            
            # Predict densities using normalized coordinates
            normalized_densities = predictor.predict_density_batched(grid_points, z_vector)
            
            # Check the range of normalized densities from the model
            norm_min = normalized_densities.min()
            norm_max = normalized_densities.max()
            print(f"Raw model output range: [{norm_min:.4f}, {norm_max:.4f}]")
            
            # Clamp values to [0, 1] if they exceed this range (shouldn’t happen with Sigmoid)
            if norm_min < 0.0 or norm_max > 1.0:
                print(f"Warning: Model outputs exceed [0, 1] range. Clamping values.")
                normalized_densities = np.clip(normalized_densities, 0.0, 1.0)
            
            # Denormalize the densities
            densities = denormalize_densities(normalized_densities, norm_params)
            
            # Place densities at integer coordinates
            volume[int_coords[:, 0], int_coords[:, 1], int_coords[:, 2]] = densities.squeeze()
            
            output_path = os.path.join(output_dir, f'{file_name}.mrc')
            with mrcfile.new(output_path, overwrite=True) as mrc:
                mrc.set_data(volume)
                mrc.update_header_from_data()
            
            print(f"Created density map at: {output_path}")
            print(f"Final volume range: {volume.min():.4f} to {volume.max():.4f}")
            
        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")
            continue
        
        finally:
            torch.cuda.empty_cache()

def main():
    volume_shape = (272, 632, 632)  # Matches your map "00256_10.00Apx.mrc"
    checkpoint_path = "/home/zcy/seperate_VAE/fresh/cri_loss_1/model_best.pth"
    output_dir = "/home/zcy/seperate_VAE/output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        create_density_map(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            volume_shape=volume_shape,
        )
        print(f"Successfully finished creating density maps in: {output_dir}")
    
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()