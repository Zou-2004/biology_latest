import torch
import numpy as np
import mrcfile
import os
from train import im_network
import h5py
from tqdm import tqdm

class DensityPredictor:
    def __init__(self, pos_checkpoint_path, neg_checkpoint_path, pos_data_path, neg_data_path, device='cuda'):
        self.device = device
        self.point_dim = 3
        self.ef_dim = 256
        self.gf_dim = 256
        self.z_dim = 8
        self.chunk_size = 100000
        
        # Load both positive and negative generators
        self.pos_generator = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim).generator.to(device)
        self.neg_generator = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim).generator.to(device)
        
        # Load checkpoints
        pos_checkpoint = torch.load(pos_checkpoint_path)
        neg_checkpoint = torch.load(neg_checkpoint_path)
        
        # Extract generator weights for both networks
        pos_generator_state_dict = {k.replace('generator.', ''): v 
                                  for k, v in pos_checkpoint['model_state_dict'].items() 
                                  if k.startswith('generator.')}
        neg_generator_state_dict = {k.replace('generator.', ''): v 
                                  for k, v in neg_checkpoint['model_state_dict'].items() 
                                  if k.startswith('generator.')}
        
        self.pos_generator.load_state_dict(pos_generator_state_dict)
        self.neg_generator.load_state_dict(neg_generator_state_dict)
        
        self.pos_generator.eval()
        self.neg_generator.eval()

        # Load normalization parameters
        with h5py.File(pos_data_path, 'r') as f:
            self.pos_min = f.attrs['min']
            self.pos_max = f.attrs['max']
        
        with h5py.File(neg_data_path, 'r') as f:
            self.neg_min = f.attrs['min']
            self.neg_max = f.attrs['max']
        
        print("\nNormalization parameters loaded:")
        print(f"Positive range: {self.pos_min:.6f} to {self.pos_max:.6f}")
        print(f"Negative range: {self.neg_min:.6f} to {self.neg_max:.6f}")

    def denormalize_density(self, normalized_values, mode='positive'):
        """Denormalize density values back to original range"""
        if mode == 'positive':
            original_values = normalized_values * (self.pos_max - self.pos_min) + self.pos_min
        else:
            original_values = normalized_values * (self.neg_max - self.neg_min) + self.neg_min
        return original_values

    def predict_density_batched(self, coords, mu, logvar, mode='positive'):
        try:
            total_points = coords.shape[0]
            predictions = np.zeros((total_points,), dtype=np.float32)
            
            z_vector = mu
            
            if len(z_vector.shape) == 2 and z_vector.size(0) > 1:
                z_vector = z_vector.mean(dim=0, keepdim=True)
            
            generator = self.pos_generator if mode == 'positive' else self.neg_generator
            
            for start_idx in range(0, total_points, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_points)
                chunk_coords = coords[start_idx:end_idx]
                
                with torch.no_grad():
                    coords_tensor = torch.from_numpy(chunk_coords).float().to(self.device)
                    if len(coords_tensor.shape) == 2:
                        coords_tensor = coords_tensor.unsqueeze(0)
                    
                    z_tensor = z_vector.to(self.device)
                    
                    # Get normalized predictions
                    chunk_pred = generator(coords_tensor, z_tensor, self.chunk_size)
                    chunk_values = chunk_pred.cpu().numpy()
                    
                    # Denormalize predictions
                    chunk_values = self.denormalize_density(chunk_values, mode)
                    
                    predictions[start_idx:end_idx] = chunk_values.squeeze()
                
                del coords_tensor, chunk_pred
                torch.cuda.empty_cache()
            
            return predictions.reshape(-1, 1)
            
        except Exception as e:
            print(f"Error in predict_density_batched ({mode}): {str(e)}")
            print(f"Input coords shape: {coords.shape}")
            print(f"Input mu shape: {mu.shape}")
            print(f"Z vector shape: {z_vector.shape}")
            raise

def create_density_map(pos_checkpoint_path, neg_checkpoint_path, pos_data_path, neg_data_path, 
                      pos_z_vectors_path, neg_z_vectors_path, output_dir, volume_shape):
    predictor = DensityPredictor(
        pos_checkpoint_path=pos_checkpoint_path,
        neg_checkpoint_path=neg_checkpoint_path,
        pos_data_path=pos_data_path,
        neg_data_path=neg_data_path
    )
    
    with h5py.File(pos_data_path, 'r') as pos_f, \
         h5py.File(neg_data_path, 'r') as neg_f, \
         h5py.File(pos_z_vectors_path, 'r') as pos_z_file, \
         h5py.File(neg_z_vectors_path, 'r') as neg_z_file:
        
        common_files = set(pos_f.keys()).intersection(set(neg_f.keys()))
        print(f"Found {len(common_files)} common files to process")
        
        for idx, file_name in tqdm(enumerate(common_files), total=len(common_files)):
            print(f"\nProcessing file: {file_name} ({idx+1}/{len(common_files)})")
            
            volume = np.zeros(volume_shape, dtype=np.float32)
            
            # Load and prepare z vectors
            pos_mu = torch.from_numpy(pos_z_file[file_name]['mu'][:]).float()
            pos_logvar = torch.from_numpy(pos_z_file[file_name]['logvar'][:]).float()
            neg_mu = torch.from_numpy(neg_z_file[file_name]['mu'][:]).float()
            neg_logvar = torch.from_numpy(neg_z_file[file_name]['logvar'][:]).float()
            
            # Print shapes for debugging
            print(f"\nZ vector shapes:")
            print(f"Positive mu shape: {pos_mu.shape}")
            print(f"Positive logvar shape: {pos_logvar.shape}")
            print(f"Negative mu shape: {neg_mu.shape}")
            print(f"Negative logvar shape: {neg_logvar.shape}")
            
            # Load points
            pos_points = pos_f[file_name]['points'][:]
            neg_points = neg_f[file_name]['points'][:]
            
            print(f"Points shapes:")
            print(f"Positive points shape: {pos_points.shape}")
            print(f"Negative points shape: {neg_points.shape}")
            
            # Process positive points
            if len(pos_points) > 0:
                for chunk_start in range(0, len(pos_points), predictor.chunk_size):
                    chunk_end = min(chunk_start + predictor.chunk_size, len(pos_points))
                    try:
                        points = pos_points[chunk_start:chunk_end]
                        density = predictor.predict_density_batched(points, pos_mu, pos_logvar, 'positive')
                        
                        scaled_points = (points * np.array(volume_shape)).astype(int)
                        scaled_points = np.clip(scaled_points, 0, np.array(volume_shape) - 1)
                        volume[scaled_points[:, 0], scaled_points[:, 1], scaled_points[:, 2]] = density.squeeze()
                        
                    except Exception as e:
                        print(f"Error processing positive chunk: {str(e)}")
                        continue
            
            # Process negative points
            if len(neg_points) > 0:
                for chunk_start in range(0, len(neg_points), predictor.chunk_size):
                    chunk_end = min(chunk_start + predictor.chunk_size, len(neg_points))
                    try:
                        points = neg_points[chunk_start:chunk_end]
                        density = predictor.predict_density_batched(points, neg_mu, neg_logvar, 'negative')
                        
                        scaled_points = (points * np.array(volume_shape)).astype(int)
                        scaled_points = np.clip(scaled_points, 0, np.array(volume_shape) - 1)
                        volume[scaled_points[:, 0], scaled_points[:, 1], scaled_points[:, 2]] = -density.squeeze()
                        
                    except Exception as e:
                        print(f"Error processing negative chunk: {str(e)}")
                        continue
            
            # Save the combined volume
            output_path = os.path.join(output_dir, f'{file_name}.mrc')
            with mrcfile.new(output_path, overwrite=True) as mrc:
                mrc.set_data(volume)
                mrc.update_header_from_data()
            
            print(f"Created density map at: {output_path}")
            print(f"Volume range: {volume.min():.4f} to {volume.max():.4f}")
            
            del volume
            torch.cuda.empty_cache()

def main():
    volume_shape = (272, 632, 632)
    pos_checkpoint_path = "/home/zcy/seperate_VAE/checkpoint/density_density_svae/IM_VAE_positive.model-14.pth"
    neg_checkpoint_path = "/home/zcy/seperate_VAE/checkpoint/density_density_svae/IM_VAE_negative.model-14.pth"
    pos_z_vectors_path = '/home/zcy/seperate_VAE/z_vector_for_VAE_positive.hdf5'
    neg_z_vectors_path = '/home/zcy/seperate_VAE/z_vector_for_VAE_negative.hdf5'
    output_dir = "/home/zcy/seperate_VAE/output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        create_density_map(
            pos_checkpoint_path=pos_checkpoint_path,
            neg_checkpoint_path=neg_checkpoint_path,
            pos_data_path="/home/zcy/seperate_VAE/positive_density_data.hdf5",
            neg_data_path="/home/zcy/seperate_VAE/negative_density_data.hdf5",
            pos_z_vectors_path=pos_z_vectors_path,
            neg_z_vectors_path=neg_z_vectors_path,
            output_dir=output_dir,
            volume_shape=volume_shape
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