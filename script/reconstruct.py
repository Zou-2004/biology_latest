import torch
import numpy as np
import mrcfile
import os
import gzip
from tqdm import tqdm
from pathlib import Path

from INR_network import INR_Network

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

class DensityPredictor:
    def __init__(self, density_checkpoint_path, binary_map_dir, device='cuda'):
        self.device = device
        self.g_dim = 256
        self.point_dim = 3
        self.density_z_dim = 64
        self.chunk_size = 10000  # Increased chunk size for faster processing
        self.default_shape = (272, 632, 632)  # ~108.5M points per file

        # Load density checkpoint
        self.density_checkpoint = torch.load(density_checkpoint_path, map_location=device)

        # Initialize density network
        self.density_network = INR_Network(
            gf_dim=self.g_dim,
            z_dim=self.density_z_dim,
            point_dim=self.point_dim
        ).to(device)

        # Load state dictionary
        self.density_network.load_state_dict(self.density_checkpoint["model_state_dict"])
        self.density_network.eval()

        # Load z-vectors
        self.density_z_vectors = self.density_checkpoint["z_vectors"]
        self.density_z_vectors = {k: v.to(device) for k, v in self.density_z_vectors.items()}

        # Binary map directory
        self.binary_map_dir = Path(binary_map_dir)

        print(f"Loaded {len(self.density_z_vectors)} density z-vectors")
        print(f"Assuming default shape for all files: {self.default_shape}")
        print(f"Binary maps will be loaded from: {self.binary_map_dir}")

    def load_binary_map(self, file_name):
        """Load and decompress the binary map from .gz file"""
        binary_map_path = self.binary_map_dir / f"{file_name}_binary_map.gz"
        if not binary_map_path.exists():
            raise FileNotFoundError(f"Binary map not found: {binary_map_path}")

        with gzip.open(binary_map_path, 'rb') as f:
            binary_map = np.frombuffer(f.read(), dtype=np.uint8).reshape(self.default_shape)

        print(f"Loaded binary map for {file_name}: shape {binary_map.shape}, {np.sum(binary_map):,} non-zero points")
        return binary_map

    def predict_density_batched(self, coords, z_vector):
        """Predict density values for given coordinates"""
        total_points = coords.shape[0]
        predictions = np.zeros(total_points, dtype=np.float32)

        if len(z_vector.shape) == 1:
            z_vector = z_vector.unsqueeze(0)  # [1, z_dim]

        with torch.no_grad():
            for start_idx in tqdm(range(0, total_points, self.chunk_size), desc="Density Prediction", leave=False):
                end_idx = min(start_idx + self.chunk_size, total_points)
                chunk_coords = coords[start_idx:end_idx]
                batch_size = chunk_coords.shape[0]

                coords_tensor = torch.from_numpy(chunk_coords).float().to(self.device)
                z_tensor = z_vector.expand(batch_size, -1).to(self.device)

                pred = self.density_network.generator(coords_tensor, z_tensor)
                predictions[start_idx:end_idx] = pred.cpu().numpy().squeeze()

                del coords_tensor, z_tensor, pred
                torch.cuda.empty_cache()

        return predictions.reshape(-1, 1)

def create_density_map(density_checkpoint_path, binary_map_dir, output_dir):
    predictor = DensityPredictor(
        density_checkpoint_path=density_checkpoint_path,
        binary_map_dir=binary_map_dir
    )

    file_names = list(predictor.density_z_vectors.keys())
    print(f"Found {len(file_names)} files to process")

    for idx, file_name in enumerate(file_names):
        print(f"\nProcessing file: {file_name} ({idx+1}/{len(file_names)})")

        volume_shape = predictor.default_shape
        total_points = volume_shape[0] * volume_shape[1] * volume_shape[2]
        print(f"Using volume shape: {volume_shape} ({total_points:,} points)")

        # Load binary map
        try:
            binary_map = predictor.load_binary_map(file_name)
        except Exception as e:
            print(f"Error loading binary map for {file_name}: {str(e)}")
            continue

        # Extract coordinates where binary map is 1
        occupied_indices = np.where(binary_map == 1)
        int_coords = np.stack(occupied_indices, axis=-1)  # [N, 3]
        occupied_count = int_coords.shape[0]
        print(f"Occupied points: {occupied_count:,} ({100 * occupied_count / total_points:.2f}%)")

        # Normalize coordinates to [0, 1]
        normalized_coords = int_coords / (np.array(volume_shape) - 1)
        print(f"Coordinate range: {normalized_coords.min():.4f} to {normalized_coords.max():.4f}")

        density_z = predictor.density_z_vectors[file_name]

        try:
            # Predict density values for occupied points only
            print("Predicting density values...")
            density_values = predictor.predict_density_batched(normalized_coords, density_z)

            # Create volume
            density_volume = np.zeros(volume_shape, dtype=np.float32)
            density_volume[occupied_indices] = density_values.squeeze()

            # Save density map
            output_path = os.path.join(output_dir, f'{file_name}_reconstructed.mrc')
            with mrcfile.new(output_path, overwrite=True) as mrc:
                mrc.set_data(density_volume)
                mrc.update_header_from_data()

            print(f"Created density map at: {output_path}")
            print(f"Density volume range: {density_volume.min():.4f} to {density_volume.max():.4f}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

        del binary_map, int_coords, normalized_coords, density_values
        torch.cuda.empty_cache()

def main():
    density_checkpoint_path = "/home/zcy/seperate_VAE/script/checkpoint/density/model_best.pth"
    binary_map_dir = "/home/zcy/seperate_VAE/compressed_binary_maps"  # Directory with .gz files
    output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    try:
        create_density_map(
            density_checkpoint_path=density_checkpoint_path,
            binary_map_dir=binary_map_dir,
            output_dir=output_dir
        )
        print(f"Successfully created density maps in: {output_dir}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()