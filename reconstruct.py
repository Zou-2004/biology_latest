import torch
import numpy as np
import mrcfile
import os
from train import im_network
import h5py
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class DensityPredictor:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.point_dim = 3
        self.ef_dim = 256
        self.gf_dim = 256
        self.z_dim = 256
        self.chunk_size = 100000
        # Load network
        self.network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        # Filter the state dictionary to only include keys that are in the current model
        model_dict = self.network.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                        if k in model_dict}
        
        # Check what keys were loaded
        print(f"Successfully loaded {len(pretrained_dict)} parameters")
        print(f"Missing {len(model_dict) - len(pretrained_dict)} parameters")
        
        # Load the filtered state dictionary
        self.network.load_state_dict(pretrained_dict, strict=False)
        self.network.eval()

    def predict_density_batched(self, coords, z_vector):
        try:
            total_points = coords.shape[0]
            predictions = np.zeros((total_points,), dtype=np.float32)
            
            if len(z_vector.shape) == 2 and z_vector.size(0) > 1:
                z_vector = z_vector.mean(dim=0, keepdim=True)
            
            for start_idx in range(0, total_points, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_points)
                chunk_coords = coords[start_idx:end_idx]
                
                with torch.no_grad():
                    coords_tensor = torch.from_numpy(chunk_coords).float().to(self.device)
                    # if len(coords_tensor.shape) == 2:
                    #     coords_tensor = coords_tensor.unsqueeze(0)
                    
                    z_tensor = z_vector.to(self.device)
                    
                    # Get predictions
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

def create_density_map(checkpoint_path, z_vectors_path, output_dir, volume_shape):
    predictor = DensityPredictor(checkpoint_path=checkpoint_path)
    
    # Create integer coordinates first
    x = np.arange(volume_shape[0])
    y = np.arange(volume_shape[1])
    z = np.arange(volume_shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Store integer coordinates for later use
    int_coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # Normalize coordinates for network input [0,1]
    grid_points = int_coords / (np.array(volume_shape) - 1)
    
    print(f"Total grid points: {grid_points.shape[0]}")
    print(f"Volume shape: {volume_shape}")
    print(f"Normalized coordinate range: [{grid_points.min():.4f}, {grid_points.max():.4f}]")
    # print(f"Integer coordinate range: [{int_coords.min()}, {int_coords.max()}]")
    
    with h5py.File(z_vectors_path, 'r') as z_file:
        file_names = list(z_file.keys())
        print(f"Found {len(file_names)} files to process")
        
        for idx, file_name in tqdm(enumerate(file_names)):
            print(f"\nProcessing file: {file_name} ({idx+1}/{len(file_names)})")
            
            z_vector = torch.from_numpy(z_file[file_name][:]).float()
            print(f"Z vector shape: {z_vector.shape}")
            
            try:
                # Initialize empty volume
                volume = np.zeros(volume_shape, dtype=np.float32)
                
                # Predict densities using normalized coordinates
                densities = predictor.predict_density_batched(grid_points, z_vector)
                
                # Place densities at integer coordinates
                volume[int_coords[:, 0], int_coords[:, 1], int_coords[:, 2]] = densities.squeeze()
                
                output_path = os.path.join(output_dir, f'{file_name}.mrc')
                with mrcfile.new(output_path, overwrite=True) as mrc:
                    mrc.set_data(volume)
                    # mrc.voxel_size = spacing
                    mrc.update_header_from_data()
                
                print(f"Created density map at: {output_path}")
                print(f"Volume range: {volume.min():.4f} to {volume.max():.4f}")
                print(f"Volume shape: {volume.shape}")
                
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")
                continue
            
            finally:
                torch.cuda.empty_cache()

def main():
    volume_shape = (272, 632, 632)
    # voxel_spacing = (1.0, 1.0, 1.0)  # Voxel spacing in Angstroms 
    checkpoint_path = "/home/zcy/seperate_VAE/checkpoint/density_density_ae/model_best.pth"
    # checkpoint_path = "/home/zcy/seperate_VAE/checkpoint/density_density_vae/best_IM_VAE.model.pth"
    z_vectors_path = '/home/zcy/seperate_VAE/checkpoint/density_density_ae/best_z_vectors.hdf5'
    output_dir = "/home/zcy/seperate_VAE/output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        create_density_map(
            checkpoint_path=checkpoint_path,
            z_vectors_path=z_vectors_path,
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

if __name__ =="__main__":
    main()