import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import numpy as np
import h5py
from INR_network import *

class IM_AE(object):
    def __init__(self, config):
        # DDP setup
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.device_id = self.rank % torch.cuda.device_count()

        # Model configuration
        self.max_to_keep = 1
        self.point_dim = 3
        self.gf_dim = 256
        self.chunk_size = 50000
        self.z_dim =64
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.grad_clip = 5.0
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.dataset_name)
        self.z_file_path = os.path.join(self.checkpoint_path,"random_z_vectors.hdf5")
        if self.rank == 0:
            # Make sure directory exists
            os.makedirs(os.path.dirname(self.z_file_path), exist_ok=True)
        # Get number of files and initialize learnable z vectors
        with h5py.File(os.path.join(self.data_dir, 'density_data.hdf5'), 'r') as f:
            self.num_files = len(f.keys())
            file_names = list(f.keys())
            if self.rank == 0:
                print(f"Total number of files: {self.num_files}")
                if not os.path.exists(self.z_file_path):
                    # Initialize z vectors as random tensors
                    self.fixed_file_z_data = {fname: torch.randn(self.z_dim) for fname in file_names}
                    with h5py.File(self.z_file_path, 'w') as zf:
                        for fname, z in self.fixed_file_z_data.items():
                            zf.create_dataset(fname, data=z.cpu().numpy())
                    if os.path.exists(self.z_file_path):
                        print("Generated and saved new fixed z vectors to", self.z_file_path)
                        print("Fixed z vectors:", {k: v[:5].tolist() for k, v in self.fixed_file_z_data.items()})
                    else:
                        raise RuntimeError(f"Failed to save fixed z vectors to {self.z_file_path}")
                else:
                    with h5py.File(self.z_file_path, 'r') as zf:
                        zf_keys = set(zf.keys())
                        missing_keys = [fname for fname in file_names if fname not in zf_keys]
                        if missing_keys:
                            raise KeyError(f"Missing keys in {self.z_file_path}: {missing_keys}")
                        self.fixed_file_z_data = {fname: torch.tensor(zf[fname][:]) for fname in file_names}
                    print("Loaded fixed z vectors from", self.z_file_path)
                    # print("Fixed z vectors:", {k: v[:5].tolist() for k, v in self.fixed_file_z_data.items()})
            else:
                self.fixed_file_z_data = {fname: None for fname in file_names}  # Initialize with None

            # Broadcast fixed_file_z_data explicitly
            if self.rank == 0:
                broadcast_data = [(fname, z) for fname, z in self.fixed_file_z_data.items()]
            else:
                broadcast_data = [(fname, None) for fname in file_names]
            dist.barrier()
            dist.broadcast_object_list(broadcast_data, src=0)
            self.fixed_file_z_data = {fname: z for fname, z in broadcast_data}
            
            # Create learnable Parameter objects for each z vector
            self.learnable_z = {}
            for fname, z_data in self.fixed_file_z_data.items():
                # Convert to Parameter and move to device
                self.learnable_z[fname] = torch.nn.Parameter(z_data.clone().to(self.device_id))

        # Initialize network
        self.network = INR_Network(self.gf_dim, self.z_dim, self.point_dim).to(self.device_id)
        self.network = DDP(
            self.network,
            device_ids=[self.device_id],
            find_unused_parameters=False,
            broadcast_buffers=True
        )

        # Setup optimizer with separate parameter groups
        world_size = dist.get_world_size()
        self.scaled_lr = config.learning_rate * world_size
        
        # Create parameter list for z vectors (much smaller learning rate)
        z_params = []
        for z_param in self.learnable_z.values():
            z_params.append(z_param)
        
        # Setup optimizer with different learning rates
        self.optimizer = optim.Adam([
            {'params': self.network.parameters(), 'lr': self.scaled_lr},
            {'params': z_params, 'lr': self.scaled_lr * 0.1}  # Very small learning rate for z
        ])
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.999)
        self.checkpoint_name = 'IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

    def load_checkpoint(self, checkpoint_path):
        """Load model, optimizer, and z vectors from a checkpoint."""
        if self.rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.device_id}')
        self.network.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load epoch and loss
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        best_loss = checkpoint['loss']
        
        # Load z vectors if they exist
        if 'z_vectors' in checkpoint:
            z_vectors = checkpoint['z_vectors']
            for fname, z in z_vectors.items():
                if fname in self.learnable_z:
                    # Update the parameter with saved value
                    self.learnable_z[fname].data.copy_(z)
        
        if self.rank == 0:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {best_loss:.6f}")
        
        return start_epoch, best_loss

    def update_learning_rate(self, new_lr):
        """Update the optimizer's learning rate."""
        self.scaled_lr = new_lr  # Base LR per GPU
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i == 0:  # First parameter group (network)
                param_group['lr'] = new_lr
            else:  # Second parameter group (z vectors)
                param_group['lr'] = new_lr * 0.01  # Keep the same ratio
        if self.rank == 0:
            print(f"Updated base LR to {new_lr:.6f}, z vector LR to {new_lr * 0.01:.8f}")

    def combined_loss(self, predicted_density, true_density):
        """Compute a weighted MSE loss on the unnormalized data."""
        point_wise_mse = F.mse_loss(predicted_density, true_density, reduction='none')
        
        # Value-based weights (focus on larger density values)
        value_weights = (true_density.abs() + 1e-4) / (true_density.abs().mean() + 1e-4)
        value_weights = value_weights**1.5  # Reduced exponent from 2 to 1.5 for smoother weighting
        
        # Error-based weights (focus on high-error regions)
        error_weights = torch.ones_like(point_wise_mse)
        high_error_mask = point_wise_mse > torch.quantile(point_wise_mse, 0.90)
        error_weights[high_error_mask] = 2.0  # Reduced weight from 3.0 to 2.0 for high-error regions
        
        # Combine weights
        weights = value_weights * error_weights
        weights = weights / weights.mean()  # Normalize
        weights = torch.clamp(weights, max=20.0)  # Reduced max weight from 50.0 to 20.0
        weights = weights / weights.mean()  # Re-normalize after clamping
        
        # Apply weights to MSE
        weighted_mse = (point_wise_mse * weights).mean()
        return weighted_mse

    # def combined_loss(self, predicted_density, true_density):
    #     # L2 loss (mean squared error)
    #     L2_loss = F.mse_loss(predicted_density, true_density)
        
    #     # Log loss: mean absolute log difference
    #     log_pred = torch.log(torch.abs(predicted_density) + self.epsilon)
    #     log_true = torch.log(torch.abs(true_density) + self.epsilon)
    #     log_loss = torch.mean(torch.abs(log_pred - log_true))
        
    #     # Combined loss with weight
    #     scaled_log_loss = self.log_weight * log_loss
    #     combine_loss = scaled_log_loss + L2_loss
        
    #     return scaled_log_loss, L2_loss, combine_loss
    
    # def combined_loss(self, predicted_density, true_density):
    #     """Compute the combined loss."""
    #     scaled_L1_loss=0.2*F.l1_loss(predicted_density, true_density)
    #     scaled_L2_loss=F.mse_loss(predicted_density, true_density)
    #     combine_loss=scaled_L1_loss+scaled_L2_loss
    #     return scaled_L1_loss, scaled_L2_loss, combine_loss

    # def combine_loss(self, predicted_density, true_density):
    #     L2_loss=F.mse_loss(predicted_density, true_density)
    #     return L2_loss
    
    @property
    def model_dir(self):
        return f"{self.dataset_name}_density_ae"
    
    def train(self, config):
        if self.rank == 0:
            print(f"Start running DDP training on rank {self.rank}.")
            print(f"\n----------Training Summary----------")
            print(f"Z vector dimension: {self.z_dim}")
            print(f"Chunk size: {self.chunk_size}")
            print(f"Total epochs: {config.epoch}")
            print(f"Network learning rate: {self.scaled_lr}")
            print(f"Z vectors learning rate: {self.scaled_lr * 0.01}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            total_params = sum(p.numel() for p in self.network.parameters())
            trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            z_params = sum(p.numel() for fname, p in self.learnable_z.items())
            print(f"\nModel Statistics:")
            print(f"Total network parameters: {total_params:,}")
            print(f"Trainable network parameters: {trainable_params:,}")
            print(f"Z vector parameters: {z_params:,}")
            print("-" * 40 + "\n")

        data_dir = os.path.join(self.data_dir, 'density_data.hdf5')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data file not found: {data_dir}")
        checkpoint_path = os.path.join(self.checkpoint_path, 'model_best.pth')
        best_loss = float('inf')
        start_epoch = 0
        # if config.load_checkpoint and os.path.exists(checkpoint_path):
        #     start_epoch, best_loss = self.load_checkpoint(checkpoint_path)
        #     # Update LR to config value (e.g., 0.00025 from main.py)
        #     self.update_learning_rate(self.scaled_lr)
        #     # Adjust scheduler to match resumed epoch
        #     # for _ in range(start_epoch):
        #     #     self.scheduler.step()
        for epoch in range(start_epoch, config.epoch):
            epoch_start_time = time.time()
            dist.barrier()
            
            if self.rank == 0:
                print(f"\nEpoch {epoch + 1}/{config.epoch}")
                print("=" * 40)

            self.network.train()
            epoch_loss = self.train_epoch(epoch, config, data_dir)
                
            self.scheduler.step()
            
            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                z_lr = self.optimizer.param_groups[1]['lr']
                print(f"\nEpoch Statistics:")
                print(f"Time taken: {epoch_time:.2f} seconds")
                print(f"Average loss: {epoch_loss:.6f}")
                print(f"Network learning rate: {current_lr:.6e}")
                # print(f"Z vectors learning rate: {z_lr:.8e}")
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.save_checkpoint(epoch, epoch_loss, is_best=True)
                    print(f"New best model saved! Loss: {epoch_loss:.6f}")
                if (epoch + 1) % 10 == 0:
                    print(f"\nTraining Progress: {(epoch + 1)/config.epoch*100:.1f}%")
                    print(f"Best loss so far: {best_loss:.6f}")

            dist.barrier()

        dist.barrier()
        dist.destroy_process_group()
        if self.rank == 0:
            print("\nTraining completed!")
            print(f"Best model loss: {best_loss:.6f}")
            print(f"Model saved at: {self.checkpoint_path}")

    def train_epoch(self, epoch, config, data_file):
        epoch_start_time = time.time()
        world_size = dist.get_world_size()

        with h5py.File(data_file, 'r') as f:
            file_names = list(f.keys())
            if self.rank == 0:
                np.random.shuffle(file_names)
            dist.broadcast_object_list([file_names], src=0)
            epoch_total_loss = 0.0
            epoch_total_samples = 0
            
            for file_idx, file_name in enumerate(file_names):
                if self.rank == 0:
                    print(f"\nProcessing file {file_idx + 1}/{len(file_names)}: {file_name}")
                    file_start_time = time.time()

                points_data = f[file_name]['points']
                values_data = f[file_name]['values']
                total_points = len(points_data)
                
                # Get the learnable z parameter for this file
                file_z = self.learnable_z[file_name]  # Already a Parameter on the correct device

                points_per_gpu = total_points // world_size
                remainder = total_points % world_size
                start_idx = self.rank * points_per_gpu + min(self.rank, remainder)
                end_idx = start_idx + points_per_gpu + (1 if self.rank < remainder else 0)
                local_total_points = end_idx - start_idx

                self.network.train()
                file_total_loss = 0.0
                file_processed_samples = 0
                # max_loss = 0.0

                for chunk_start in range(0, local_total_points, self.chunk_size):
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    chunk_end = min(chunk_start + self.chunk_size, local_total_points)
                    current_batch_size = chunk_end - chunk_start

                    points_chunk = torch.from_numpy(points_data[start_idx + chunk_start:start_idx + chunk_end]).float().to(self.device_id)
                    values_chunk = torch.from_numpy(values_data[start_idx + chunk_start:start_idx + chunk_end]).float().to(self.device_id)
                    if len(values_chunk.shape) == 1:
                        values_chunk = values_chunk.unsqueeze(-1)

                    predicted_densities = self.network.module.generator(points_chunk, file_z)
                    # L1,L2,recon_loss = self.combined_loss(predicted_densities, values_chunk)
                    recon_loss = self.combined_loss(predicted_densities, values_chunk)
                    loss = recon_loss
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                    for param in self.network.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data)
                            param.grad.data /= world_size
                    
                    # Also handle gradients for z vectors
                    if file_z.grad is not None:
                        dist.all_reduce(file_z.grad.data)
                        file_z.grad.data /= world_size
                    
                    self.optimizer.step()

                    with torch.no_grad():
                        local_loss = loss.item()
                        loss_value = local_loss * current_batch_size
                        stats = torch.tensor([loss_value, current_batch_size], device=self.device_id)
                        dist.all_reduce(stats)
                        chunk_total_loss = stats[0].item()
                        chunk_total_samples = stats[1].item()
                        file_total_loss += chunk_total_loss
                        file_processed_samples += chunk_total_samples
                        # max_loss = max(max_loss, local_loss)

                        if self.rank == 0 and chunk_start >= local_total_points // 2 and not hasattr(self, f'logged_{file_name}'):
                            # progress = chunk_start / local_total_points * 100
                            running_avg = file_total_loss / file_processed_samples
                            # print(f"Progress: 50% | Chunk Loss: {local_loss:.6f} |L1_loss: {L1:.6f} | L2_loss: {L2:.6f} | Running Avg: {running_avg:.6f}")
                            print(f"Progress: 50% | Chunk Loss: {local_loss:.6f} | Running Avg: {running_avg:.6f}")
                            setattr(self, f'logged_{file_name}', True)

                    del points_chunk, values_chunk, predicted_densities, loss
                    torch.cuda.empty_cache()

                if self.rank == 0:
                    file_avg_loss = file_total_loss / file_processed_samples
                    print(f"File {file_name} completed: Avg Loss: {file_avg_loss:.6f}")
                    delattr(self, f'logged_{file_name}') if hasattr(self, f'logged_{file_name}') else None

                epoch_total_loss += file_total_loss
                epoch_total_samples += file_processed_samples

            epoch_stats = torch.tensor([epoch_total_loss, epoch_total_samples], device=self.device_id)
            dist.all_reduce(epoch_stats)
            epoch_avg_loss = epoch_stats[0].item() / epoch_stats[1].item()

            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch Summary: Average Loss: {epoch_avg_loss:.6f}, Total Time: {epoch_time:.2f} seconds")
            return epoch_avg_loss

    def save_checkpoint(self, epoch, loss, is_best=False):
        if self.rank == 0:
            os.makedirs(self.checkpoint_path, exist_ok=True)
            
            # Gather all optimized z vectors
            z_vectors_dict = {}
            for fname, z_param in self.learnable_z.items():
                z_vectors_dict[fname] = z_param.detach().cpu()
            
            if is_best:
                # Save z vectors to the checkpoint
                best_model = {
                    'model_state_dict': self.network.module.state_dict() if isinstance(self.network, DDP) else self.network.state_dict(),
                    'epoch': epoch,
                    'loss': loss,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'z_vectors': z_vectors_dict  # Include optimized z vectors
                }
                torch.save(best_model, os.path.join(self.checkpoint_path, 'model_best.pth'))
                
                # # Also save z vectors to a separate HDF5 file for easier access
                # with h5py.File(os.path.join(self.checkpoint_path, 'best_z_vectors.hdf5'), 'w') as f:
                #     for fname, z in z_vectors_dict.items():
                #         f.create_dataset(fname, data=z.numpy())
                
                print(f"Saved new best model and optimized latent vector at epoch {epoch} with loss {loss:.6f}")
                # print(f"Optimized z vectors saved to {os.path.join(self.checkpoint_path, 'best_z_vectors.hdf5')}")