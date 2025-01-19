import os
import time
import numpy as np
import h5py
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# import math
from VAE_network import *

class IM_VAE(object):
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_to_keep = 1
        self.point_dim = 3
        self.ef_dim = 256
        self.gf_dim = 256
        self.chunk_size = 100000
        self.z_dim = 256
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.grad_clip = 5.0
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        
        # Single network for both positive and negative values
        self.network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)
        self.network.to(self.device)
        
        initial_lr = config.learning_rate
        
        # Single optimizer
        self.optimizer = optim.AdamW([
            {'params': self.network.parameters(), 'initial_lr': initial_lr}
        ], lr=initial_lr, betas=(config.beta1, 0.999))

        # Single scheduler
        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=30, gamma=0.1,
        #     last_epoch=config.epoch-1 if hasattr(config, 'epoch') else -1
        # )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100,
            last_epoch=config.epoch-1 if hasattr(config, 'epoch') else -1)
        
        
        self.checkpoint_name = 'IM_VAE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0
        
        self.beta = 0.01  # KL divergence weight


    # def combined_loss(self, predicted_density, true_density):
    #     # Basic losses
    #     point_wise_mse = F.mse_loss(predicted_density, true_density, reduction='none')
        
    #     # Safe relative error calculation
    #     relative_error = point_wise_mse / (true_density.abs() + 1e-4)
    #     loss_threshold = torch.quantile(relative_error, 0.75)
        
    #     # Soft weighting for high-error points
    #     weights = 1.0 + torch.sigmoid(10 * (relative_error - loss_threshold))
        
    #     # Weighted losses
    #     weighted_mse = (point_wise_mse * weights).mean()
    #     total_loss = weighted_mse
    #     return total_loss

    def unmodified_loss(self, predicted_density, true_density):
       point_wise_mse = F.mse_loss(predicted_density, true_density, reduction='none')
       return point_wise_mse.mean()

    # def combined_loss(self, predicted_density, true_density):

    #     point_wise_mse = F.mse_loss(predicted_density, true_density, reduction='none')
        
    #     # Value-based importance
    #     value_weights = (true_density.abs() + 1e-4) / (true_density.abs().mean() + 1e-4)
        
    #     # Error-based importance
    #     error_weights = torch.ones_like(point_wise_mse)
    #     high_error_mask = point_wise_mse > torch.quantile(point_wise_mse, 0.5)
    #     error_weights[high_error_mask] = 2.0  # Double weight for high errors
        
    #     # Combined weighting
    #     weights = value_weights * error_weights
    #     weighted_mse = (point_wise_mse * weights).mean()
        
    #     return weighted_mse

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

        # if os.path.exists(z_file_path):
        #     os.remove(z_file_path)
       

        for epoch in range(start_epoch, config.epoch):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{config.epoch}")
            print("=" * 40)

            # Determine whether to save or load z vectors
            save_z = not (os.path.exists(z_file_path))  # Save z vectors only in the first epoch
            use_precomputed_z = os.path.exists(z_file_path)

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
                            # print(points_chunk.shape)
                            self.optimizer.zero_grad()
                            predicted_densities = self.network.generator(points_chunk, z_avg)
                            # print(predicted_densities.shape, values_chunk.shape)
                            # exit()
                            if predicted_densities.shape != values_chunk.shape:
                                raise Exception("shape of generator and ground truth not matched!")
                
                            # predicted_densities = predicted_densities.contiguous().view(-1, 1)
                            # values_chunk = values_chunk.contiguous().view(-1, 1)
                            
                            # loss = self.combined_loss(predicted_densities, values_chunk)
                            loss = self.unmodified_loss(predicted_densities, values_chunk)
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
