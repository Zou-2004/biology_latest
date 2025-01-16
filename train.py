import os
import time
import numpy as np
import h5py
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import math



def positional_encoding(points, num_frequencies, include_input=True, log_sampling=True):
    """
    Apply positional encoding to 3D coordinates (NeRF-style).

    Args:
        points: Tensor of shape [B, N, 3], where B is batch size, N is the number of points, and 3 is (x, y, z).
        num_frequencies: Number of frequency bands for encoding.
        include_input: Whether to include the original coordinates in the output.
        log_sampling: Whether to use logarithmic frequency sampling (default: True).

    Returns:
        Tensor of shape [B, N, 3 * (2 * num_frequencies) + (3 if include_input else 0)].
    """
    # Determine frequency bands
    if log_sampling:
        frequencies = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies).to(points.device)
    else:
        frequencies = torch.linspace(1.0, 2. ** (num_frequencies - 1), num_frequencies).to(points.device)

    # Compute positional encodings
    encoded = []
    if include_input:
        encoded.append(points)  # Include the original coordinates

    for freq in frequencies:
        encoded.append(torch.sin(points * freq * math.pi))  # Scale by π for NeRF-style encoding
        encoded.append(torch.cos(points * freq * math.pi))

    # Concatenate original coordinates and encoded features
    return torch.cat(encoded, dim=-1)

class generator(nn.Module):
    def __init__(self, z_dim, point_dim, gf_dim, num_frequencies=10, include_input=True):
        super(generator, self).__init__()
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.gf_dim = gf_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input  # Include raw input coordinates in PE

        # Update the input dimension based on positional encoding
        self.encoded_dim = self.point_dim * (2 * self.num_frequencies)  # Sin and cos
        if self.include_input:
            self.encoded_dim += self.point_dim  # Add raw input coordinates

        self.input_dim = self.z_dim + self.encoded_dim

        # Define network layers
        self.linear_1 = nn.Linear(self.input_dim, self.gf_dim * 8, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim * 8, self.gf_dim * 4, bias=True)
        self.linear_5 = nn.Linear(self.gf_dim * 4, self.gf_dim * 2, bias=True)
        self.linear_6 = nn.Linear(self.gf_dim * 2, self.gf_dim * 1, bias=True)
        self.linear_7 = nn.Linear(self.gf_dim * 1, 1, bias=True)

        # Initialize weights
        for layer in [self.linear_1, self.linear_2, self.linear_3,
                      self.linear_4, self.linear_5, self.linear_6]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.05)
            nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.linear_7.weight, mean=0.0, std=0.05)
        nn.init.constant_(self.linear_7.bias, 0)

    def forward(self, points, z, chunk_size):
        # Ensure consistent shapes
        if len(points.shape) == 2:
            points = points.unsqueeze(0)  # [1, N, 3]

        batch_size = points.size(0)
        num_points = points.size(1)

        # Apply positional encoding to points
        points_encoded = positional_encoding(points, self.num_frequencies, include_input=self.include_input)  # [B, N, encoded_dim]

        # Ensure z has the correct shape
        if len(z.shape) == 1:
            z = z.unsqueeze(0)  # [1, z_dim]
        elif len(z.shape) == 2 and z.size(0) != batch_size:
            z = z.repeat(batch_size, 1)

        # Expand z to match points
        zs = z.unsqueeze(1).expand(batch_size, num_points, -1)

        # Concatenate encoded points and z
        pointz = torch.cat([points_encoded, zs], dim=-1)  # [B, N, input_dim]

        # Process in chunks
        outputs = []
        for i in range(0, num_points, chunk_size):
            end_idx = min(i + chunk_size, num_points)
            chunk = pointz[:, i:end_idx, :]

            # Maintain batch dimension
            chunk_flat = chunk.reshape(-1, self.input_dim)

            l1 = F.leaky_relu(self.linear_1(chunk_flat), negative_slope=0.02)
            l2 = F.leaky_relu(self.linear_2(l1), negative_slope=0.02)
            l3 = F.leaky_relu(self.linear_3(l2), negative_slope=0.02)
            l4 = F.leaky_relu(self.linear_4(l3), negative_slope=0.02)
            l5 = F.leaky_relu(self.linear_5(l4), negative_slope=0.02)
            l6 = F.leaky_relu(self.linear_6(l5), negative_slope=0.02)
            l7 = self.linear_7(l6)

            # Reshape back to batch dimension
            chunk_output = l7.view(batch_size, end_idx - i, 1)
            outputs.append(chunk_output)

        # Combine chunks
        density = torch.cat(outputs, dim=1)  # [B, N, 1]

        return density.squeeze(-1)  # Returns [B, N]

class encoder(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(encoder, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        
        self.point_net = nn.Sequential(
            nn.Linear(3, self.ef_dim),
            nn.LayerNorm(self.ef_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(self.ef_dim, self.ef_dim*2),
            nn.LayerNorm(self.ef_dim*2),
            nn.LeakyReLU(0.02),
            nn.Linear(self.ef_dim*2, self.ef_dim*4),
            nn.LayerNorm(self.ef_dim*4),
            nn.LeakyReLU(0.02)
        )
        
        # Separate layers for mean and log variance
        self.fc_mu = nn.Sequential(
            nn.Linear(self.ef_dim*4, self.ef_dim*8),
            nn.LayerNorm(self.ef_dim*8),
            nn.LeakyReLU(0.02),
            nn.Linear(self.ef_dim*8, self.z_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.ef_dim*4, self.ef_dim*8),
            nn.LayerNorm(self.ef_dim*8),
            nn.LeakyReLU(0.02),
            nn.Linear(self.ef_dim*8, self.z_dim)
        )
    
    def forward(self, points, is_training=False):
        if len(points.shape) == 2:
            points = points.unsqueeze(0)
        
        B, N, C = points.shape
        assert C == 3, f"Expected points to have 3 coordinates, got {C}"
        
        points_flat = points.reshape(-1, 3)
        point_features = self.point_net(points_flat)
        point_features = point_features.reshape(B, N, -1)
        
        global_features = torch.max(point_features, dim=1)[0]
        
        mu = self.fc_mu(global_features)
        logvar = self.fc_logvar(global_features)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return mu, logvar, z

class im_network(nn.Module):
    def __init__(self, ef_dim, gf_dim, z_dim, point_dim):
        super(im_network, self).__init__()
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.encoder = encoder(self.ef_dim, self.z_dim)
        self.generator = generator(self.z_dim, self.point_dim, self.gf_dim)


class IM_VAE(object):
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_to_keep = 1
        self.point_dim = 3
        self.ef_dim = 256
        self.gf_dim = 256
        self.z_dim = 32
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.grad_clip = 5.0
        self.chunk_size = 100000
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        
        # Single network for both positive and negative values
        self.network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)
        self.network.to(self.device)
        
        initial_lr = config.learning_rate
        
        # Single optimizer
        self.optimizer = optim.Adam([
            {'params': self.network.parameters(), 'initial_lr': initial_lr}
        ], lr=initial_lr, betas=(config.beta1, 0.999))

        # Single scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1,
            last_epoch=config.epoch-1 if hasattr(config, 'epoch') else -1
        )

        self.checkpoint_name = 'IM_VAE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0
        
        self.beta = 0.01  # KL divergence weight


    def combined_loss(self, predicted_density, true_density, mu, logvar):
        # Basic losses
        point_wise_mse = F.mse_loss(predicted_density, true_density, reduction='none')
        # point_wise_l1 = F.l1_loss(predicted_density, true_density, reduction='none')
        
        # Safe relative error calculation
        relative_error = point_wise_mse / (true_density.abs() + 1e-4)
        loss_threshold = torch.quantile(relative_error, 0.5)
        
        # Soft weighting for high-error points
        weights = 1.0 + torch.sigmoid(10 * (relative_error - loss_threshold))
        
        # Weighted losses
        weighted_mse = (point_wise_mse * weights).mean()
        # weighted_l1 = (point_wise_l1 * weights).mean()
        
        # # First-order and second-order gradient losses
        # first_derivative = torch.abs(predicted_density[1:] - predicted_density[:-1])
        # second_derivative = torch.abs(predicted_density[2:] - 2 * predicted_density[1:-1] + predicted_density[:-2])
        # first_gradient_loss = torch.mean(first_derivative)
        # second_gradient_loss = torch.mean(second_derivative)
        # gradient_loss = 1000 * (first_gradient_loss + second_gradient_loss)
        
        # Combine losses
        # recon_loss = weighted_mse + gradient_loss
        
        # exit()
        # KL divergence
        # kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print(weighted_mse.item(), gradient_loss.item())
        # # Total loss
        # total_loss = recon_loss + 0.1 * kl_divergence
        total_loss = weighted_mse
        return total_loss


    @property
    def model_dir(self):
        return "{}_density_vae".format(self.dataset_name)
    
    def train(self, config):
        data_dir = os.path.join(self.data_dir, 'density_data.hdf5')
        z_file_path = "/home/zcy/seperate_VAE/z_vectors.hdf5"
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data file not found: {data_dir}")

        best_loss = float('inf')
        start_epoch = 0  # Modify this if you implement checkpoint loading

        print(f"\n----------Training Summary----------")
        print(f"Z vector dimension: {self.z_dim}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Total epochs: {config.epoch}")
        print(f"Learning rate: {config.learning_rate}")
        print("-" * 40 + "\n")

        for epoch in range(start_epoch, config.epoch):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{config.epoch}")
            print("=" * 40)

            # Determine whether to save or load z vectors
            save_z = (epoch == 0)  # Save z vectors only in the first epoch
            use_precomputed_z = os.path.exists(z_file_path) and not save_z

            # Train for one epoch
            epoch_loss = self.train_epoch(epoch, config, data_dir, z_file_path, save_z=save_z, use_precomputed_z=use_precomputed_z)

            # Step scheduler
            self.scheduler.step()

            # Save checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_checkpoint(epoch, epoch_loss, is_best=True)
                print(f"\nNew best model! Loss: {epoch_loss:.6f}")
            else:
                self.save_checkpoint(epoch, epoch_loss, is_best=False)

            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch completed in {epoch_time:.2f}s")
            print(f"Current best loss: {best_loss:.6f}")


    def train_epoch(self, epoch, config, data_file, z_file, save_z=False, use_precomputed_z=False):
        epoch_loss_sum = 0
        total_batches = 0

        # Open the data file
        with h5py.File(data_file, 'r') as f:
            file_names = list(f.keys())
            np.random.shuffle(file_names)
            total_files = len(file_names)

            # Open z_vectors.hdf5 for reading or writing
            with h5py.File(z_file, 'a' if save_z else 'r') as z_f:
                for file_idx, file_name in enumerate(file_names):
                    try:
                        print(f"\nProcessing file {file_idx + 1}/{total_files}: {file_name}")
                        points = f[file_name]['points'][:]
                        values = f[file_name]['values'][:]
                        total_points = len(points)

                        if use_precomputed_z:
                            # Load precomputed z, mu, and logvar
                            z_avg = torch.from_numpy(z_f[file_name]['z'][:]).float().to(self.device)
                            mu_avg = torch.from_numpy(z_f[file_name]['mu'][:]).float().to(self.device)
                            logvar_avg = torch.from_numpy(z_f[file_name]['logvar'][:]).float().to(self.device)
                        else:
                            # Compute z, mu, and logvar using the encoder
                            self.network.encoder.eval()
                            z_sum, mu_sum, logvar_sum = None, None, None
                            num_chunks = 0

                            print("Computing latent vectors...")
                            for chunk_start in range(0, total_points, self.chunk_size):
                                chunk_end = min(chunk_start + self.chunk_size, total_points)
                                points_chunk = torch.from_numpy(points[chunk_start:chunk_end]).float().to(self.device)

                                with torch.no_grad():
                                    mu, logvar, z = self.network.encoder(points_chunk)

                                    # Accumulate sums
                                    if z_sum is None:
                                        z_sum = z.clone()
                                        mu_sum = mu.clone()
                                        logvar_sum = logvar.clone()
                                    else:
                                        z_sum += z
                                        mu_sum += mu
                                        logvar_sum += logvar

                                    num_chunks += 1

                                del points_chunk, mu, logvar, z
                                torch.cuda.empty_cache()

                            # Compute averages
                            z_avg = z_sum / num_chunks
                            mu_avg = mu_sum / num_chunks
                            logvar_avg = logvar_sum / num_chunks

                            if save_z:
                                # Save z, mu, and logvar to z_vectors.hdf5
                                group = z_f.create_group(file_name)
                                group.create_dataset('z', data=z_avg.cpu().numpy(), compression='gzip')
                                group.create_dataset('mu', data=mu_avg.cpu().numpy(), compression='gzip')
                                group.create_dataset('logvar', data=logvar_avg.cpu().numpy(), compression='gzip')

                            del z_sum, mu_sum, logvar_sum
                            torch.cuda.empty_cache()

                        # Train the generator using z_avg
                        self.network.train()
                        print("\nTraining generator...")
                        file_loss_sum = 0
                        file_batches = 0

                        for chunk_start in range(0, total_points, self.chunk_size):
                            chunk_end = min(chunk_start + self.chunk_size, total_points)

                            points_chunk = torch.from_numpy(points[chunk_start:chunk_end]).float().to(self.device)
                            values_chunk = torch.from_numpy(values[chunk_start:chunk_end]).float().to(self.device)

                            self.optimizer.zero_grad()
                            predicted_densities = self.network.generator(points_chunk, z_avg, chunk_end - chunk_start)

                            predicted_densities = predicted_densities.contiguous().view(-1, 1)
                            values_chunk = values_chunk.contiguous().view(-1, 1)

                            loss = self.combined_loss(predicted_densities, values_chunk, mu_avg, logvar_avg)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                            self.optimizer.step()

                            current_loss = loss.item()
                            epoch_loss_sum += current_loss
                            total_batches += 1
                            file_loss_sum += current_loss
                            file_batches += 1

                            if (chunk_start // self.chunk_size) % 5 == 0:
                                progress = (chunk_start + self.chunk_size) / total_points * 100
                                avg_loss = file_loss_sum / file_batches
                                print(f"Generator progress: {progress:.1f}% - "
                                    f"Current Loss: {current_loss:.6f} - "
                                    f"Avg Loss: {avg_loss:.6f}")

                            del points_chunk, values_chunk, predicted_densities, loss
                            torch.cuda.empty_cache()

                        if file_batches > 0:
                            file_avg_loss = file_loss_sum / file_batches
                            print(f"File {file_name} completed - Average Loss: {file_avg_loss:.6f}")

                        del z_avg, mu_avg, logvar_avg
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
                        continue

        return epoch_loss_sum / total_batches if total_batches > 0 else float('inf')

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save checkpoint with optional best model saving."""
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        # Save numbered checkpoint
        numbered_path = os.path.join(self.checkpoint_path, f"{self.checkpoint_name}-{epoch}.pth")
        torch.save(checkpoint, numbered_path)
        print(f"Saved checkpoint: {numbered_path}")
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_path = os.path.join(self.checkpoint_path, f"best_{self.checkpoint_name}.pth")
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")
        
        # Manage old checkpoints
        self._manage_old_checkpoints()

    def _manage_old_checkpoints(self):
        """Remove old checkpoints keeping only max_to_keep recent ones."""
        checkpoint_prefix = os.path.join(self.checkpoint_path, f"{self.checkpoint_name}-")
        existing_checkpoints = []
        
        for filename in os.listdir(self.checkpoint_path):
            if filename.startswith(self.checkpoint_name) and filename.endswith(".pth") and "best" not in filename:
                checkpoint_path = os.path.join(self.checkpoint_path, filename)
                epoch_num = int(filename.split('-')[-1].split('.')[0])
                existing_checkpoints.append((epoch_num, checkpoint_path))
        
        if len(existing_checkpoints) > self.max_to_keep:
            existing_checkpoints.sort(reverse=True)  # Sort by epoch number
            for _, checkpoint_path in existing_checkpoints[self.max_to_keep:]:
                try:
                    os.remove(checkpoint_path)
                    print(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    print(f"Error removing checkpoint {checkpoint_path}: {e}")

    # def load_best_checkpoint(self):
    #     """Load the best checkpoint if it exists."""
    #     best_checkpoint_path = os.path.join(self.checkpoint_path, f"best_{self.checkpoint_name}.pth")
        
    #     if os.path.exists(best_checkpoint_path):
    #         print(f"Loading best checkpoint from {best_checkpoint_path}")
    #         checkpoint = torch.load(best_checkpoint_path)
            
    #         self.network.load_state_dict(checkpoint['model_state_dict'])
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
    #         epoch = checkpoint['epoch']
    #         loss = checkpoint.get('loss', float('inf'))
    #         print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")
    #         return epoch + 1  # Return next epoch to start from
    #     else:
    #         print("No checkpoint found. Starting from scratch.")
    #         return 0

    # def save_collected_z_vectors(self, z_vectors):
    #     """Save z vectors collected during training."""
    #     z_vector_path = '/home/zcy/seperate_VAE/z_vectors.hdf5'
        
    #     if os.path.exists(z_vector_path):
    #         os.remove(z_vector_path)
        
    #     print("\nSaving z-vectors...")
    #     with h5py.File(z_vector_path, 'w') as f_out:
    #         for file_name, vectors in z_vectors.items():
    #             try:
    #                 group = f_out.create_group(file_name)
    #                 for key, value in vectors.items():
    #                     group.create_dataset(key, data=value, compression='gzip')
    #             except Exception as e:
    #                 print(f"Error saving vectors for {file_name}: {e}")
        
    #     print(f"Z-vectors saved to {z_vector_path}")