import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import numpy as np
import h5py
from VAE_network import *

class IM_VAE(object):
    def __init__(self, config):
        # DDP setup
        # print(f"Available GPUs: {torch.cuda.device_count()}")
        # print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.device_id = self.rank % torch.cuda.device_count()

        # Model configuration
        self.max_to_keep = 1
        self.point_dim = 3
        self.ef_dim = 256
        self.gf_dim = 256
        self.chunk_size = 100000
        self.z_dim = 32
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.grad_clip = 5.0
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        
        # Initialize model and move to GPU
        self.network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim).to(self.device_id)
        self.network = DDP(
            self.network, 
            device_ids=[self.device_id],
            find_unused_parameters=False, 
            broadcast_buffers=True 
        )
        world_size = dist.get_world_size()
        self.scaled_lr = config.learning_rate * math.sqrt(world_size)  # Linear scaling
        # Create optimizer with explicit initial_lr
        self.optimizer = optim.Adam([
                {'params': self.network.parameters(), 'initial_lr': self.scaled_lr}
            ], lr=self.scaled_lr)
       
        # self.optimizer = optim.Adam([
        #         {'params': self.network.parameters(), 'initial_lr': config.learning_rate}
        #     ], lr=config.learning_rate)

        # Create scheduler with initial_lr properly set
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            last_epoch=-1  # Always start from beginning when creating new scheduler
        )

        self.checkpoint_name = 'IM_VAE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0
        self.beta = 0.01

    def combined_loss(self, predicted_density, true_density):
        # Calculate MSE
        point_wise_mse = F.mse_loss(predicted_density, true_density, reduction='none')
        
        # Value-based importance weighting
        value_weights = (true_density.abs() + 1e-4) / (true_density.abs().mean() + 1e-4)
        value_weights=value_weights**2
        
        # Error-based importance weighting
        error_weights = torch.ones_like(point_wise_mse)
        high_error_mask = point_wise_mse > torch.quantile(point_wise_mse, 0.90)
        error_weights[high_error_mask] = 3.0
        
        # Combined weighting
        weights = value_weights * error_weights
        
        # Normalize weights to prevent scaling issues
        weights = weights / weights.mean()
        
        # Calculate weighted loss
        weighted_mse = (point_wise_mse * weights).mean()
        
        return weighted_mse
    
    # def unmodified_loss(self, predicted_density, true_density):
    #     return F.mse_loss(predicted_density, true_density, reduction='mean')

    def compute_z_vectors(self, points_chunk):
        with torch.no_grad():
            mu, logvar, z = self.network.module.encoder(points_chunk)
            return mu, logvar, z
        
    @property
    def model_dir(self):
        return f"{self.dataset_name}_density_vae"
    
    def train(self, config):
        if self.rank == 0:
            print(f"Start running DDP training on rank {self.rank}.")
            print(f"\n----------Training Summary----------")
            print(f"Z vector dimension: {self.z_dim}")
            print(f"Chunk size: {self.chunk_size}")
            print(f"Total epochs: {config.epoch}")
            print(f"Learning rate: {self.scaled_lr}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            # Print model statistics
            total_params = sum(p.numel() for p in self.network.parameters())
            trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            print(f"\nModel Statistics:")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Parameters per GPU: {trainable_params/torch.cuda.device_count():,}")
            print("-" * 40 + "\n")

        data_dir = os.path.join(self.data_dir, 'density_data.hdf5')
        z_file_path = "/home/zcy/seperate_VAE/z_vectors.hdf5"
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data file not found: {data_dir}")

        # Determine if we need to compute z vectors
        compute_z = False
        if self.rank == 0:
            if not os.path.exists(z_file_path):
                compute_z = True
                print("Z vector file not found. Will compute z vectors.")
            else:
                try:
                    with h5py.File(data_dir, 'r') as data_f, h5py.File(z_file_path, 'r') as z_f:
                        data_files = set(data_f.keys())
                        z_files = set(z_f.keys())
                        if data_files != z_files:
                            compute_z = True
                            os.remove(z_file_path)
                            print("Data files mismatch. Deleted old z_file, will recompute z vectors.")
                        else:
                            print("Using existing z vectors from file.")
                except Exception as e:
                    print(f"Error checking z_file status: {e}")
                    if os.path.exists(z_file_path):
                        os.remove(z_file_path)
                    compute_z = True

        # Broadcast compute_z decision to all processes
        compute_z = torch.tensor([compute_z], dtype=torch.bool, device=self.device_id)
        dist.broadcast(compute_z, src=0)
        compute_z = compute_z.item()

        # Create new z_file if needed
        if compute_z and self.rank == 0:
            with h5py.File(z_file_path, 'w') as _:
                pass
            print(f"Created new z_file at {z_file_path}")

        dist.barrier()  # Ensure z_file is created before proceeding
        
        best_loss = float('inf')
        start_epoch = 0

        try:
            for epoch in range(start_epoch, config.epoch):
                epoch_start_time = time.time()
                dist.barrier()
                
                if self.rank == 0:
                    print(f"\nEpoch {epoch + 1}/{config.epoch}")
                    print("=" * 40)

                self.network.train()
                epoch_loss = self.train_epoch(
                    epoch, 
                    config, 
                    data_dir, 
                    z_file_path,
                    compute_z=compute_z
                )
                    
                self.scheduler.step()
                
                if self.rank == 0:
                    epoch_time = time.time() - epoch_start_time
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    print(f"\nEpoch Statistics:")
                    print(f"Time taken: {epoch_time:.2f} seconds")
                    print(f"Average loss: {epoch_loss:.6f}")
                    print(f"Learning rate: {current_lr:.6e}")
                    
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        self.save_checkpoint(epoch, epoch_loss, is_best=True)
                        print(f"New best model saved! Loss: {epoch_loss:.6f}")
                    else:
                        self.save_checkpoint(epoch, epoch_loss, is_best=False)
                        print(f"Checkpoint saved. Best loss so far: {best_loss:.6f}")
                    
                    if (epoch + 1) % 10 == 0:
                        print(f"\nTraining Progress: {(epoch + 1)/config.epoch*100:.1f}%")
                        print(f"Best loss so far: {best_loss:.6f}")

                dist.barrier()

        except Exception as e:
            print(f"Rank {self.rank} encountered error: {str(e)}")
            raise e
        finally:
            try:
                dist.barrier()
                dist.destroy_process_group()
                if self.rank == 0:
                    print("\nTraining completed!")
                    print(f"Best model loss: {best_loss:.6f}")
                    print(f"Model saved at: {self.checkpoint_path}")
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")

    def train_epoch(self, epoch, config, data_file, z_file, compute_z=False):
        epoch_start_time = time.time()
        world_size = dist.get_world_size()

        with h5py.File(data_file, 'r') as f:
            file_names = list(f.keys())
            
            # Synchronize file names across processes
            if self.rank == 0:
                np.random.shuffle(file_names)
            
            # Broadcast file names from rank 0
            if self.rank == 0:
                file_names_tensor = torch.tensor([ord(c) for c in ','.join(file_names)]).to(self.device_id)
                size_tensor = torch.tensor([len(file_names_tensor)], dtype=torch.long).to(self.device_id)
            else:
                size_tensor = torch.tensor([0], dtype=torch.long).to(self.device_id)
                file_names_tensor = torch.tensor([]).to(self.device_id)
            
            dist.broadcast(size_tensor, src=0)
            if self.rank != 0:
                file_names_tensor = torch.zeros(size_tensor[0], dtype=torch.long).to(self.device_id)
            dist.broadcast(file_names_tensor, src=0)
            
            if self.rank != 0:
                file_names = ''.join([chr(i) for i in file_names_tensor.cpu().numpy()]).split(',')

            epoch_total_loss = 0.0
            epoch_total_samples = 0
            
            for file_idx, file_name in enumerate(file_names):
                if self.rank == 0:
                    print(f"\nProcessing file {file_idx + 1}/{len(file_names)}: {file_name}")

                # Load data
                points = f[file_name]['points'][:]
                values = f[file_name]['values'][:]
                total_points = len(points)

                # Distribute data across GPUs
                points_per_gpu = total_points // world_size
                remainder = total_points % world_size
                start_idx = self.rank * points_per_gpu + min(self.rank, remainder)
                end_idx = start_idx + points_per_gpu + (1 if self.rank < remainder else 0)

                local_points = points[start_idx:end_idx]
                local_values = values[start_idx:end_idx]
                local_total_points = end_idx - start_idx

                torch.cuda.synchronize()
                dist.barrier()

                # Handle z vectors
                if not compute_z:
                    # All processes read the precomputed z vectors
                    try:
                        with h5py.File(z_file, 'r') as z_f:
                            z_avg = torch.from_numpy(z_f[file_name]['z'][:]).float().to(self.device_id)
                            mu_avg = torch.from_numpy(z_f[file_name]['mu'][:]).float().to(self.device_id)
                            logvar_avg = torch.from_numpy(z_f[file_name]['logvar'][:]).float().to(self.device_id)
                    except Exception as e:
                        print(f"Rank {self.rank} error reading z_file: {e}")
                        raise
                else:
                    # In first epoch, compute and save z vectors
                    if epoch == 0:
                        if self.rank ==0:
                            print("Computing z, this will only happen in epoch 0")
                        # Compute z vectors
                        self.network.module.encoder.eval()
                        z_sums = torch.zeros((3, self.z_dim), device=self.device_id)
                        total_samples = 0

                        for chunk_start in range(0, local_total_points, self.chunk_size):
                            chunk_end = min(chunk_start + self.chunk_size, local_total_points)
                            points_chunk = torch.from_numpy(local_points[chunk_start:chunk_end]).float().to(self.device_id)
                            
                            mu, logvar, z = self.compute_z_vectors(points_chunk)
                            z_sums[0] += z.sum(dim=0)
                            z_sums[1] += mu.sum(dim=0)
                            z_sums[2] += logvar.sum(dim=0)
                            total_samples += points_chunk.size(0)

                        dist.all_reduce(z_sums)
                        total_samples_tensor = torch.tensor([total_samples], device=self.device_id)
                        dist.all_reduce(total_samples_tensor)
                        total_samples = total_samples_tensor.item()

                        z_avg = z_sums[0] / total_samples
                        mu_avg = z_sums[1] / total_samples
                        logvar_avg = z_sums[2] / total_samples

                        # Only rank 0 writes to file
                        if self.rank == 0:
                            try:
                                with h5py.File(z_file, 'a') as z_f:
                                    if file_name in z_f:
                                        del z_f[file_name]
                                    group = z_f.create_group(file_name)
                                    group.create_dataset('z', data=z_avg.cpu().numpy())
                                    group.create_dataset('mu', data=mu_avg.cpu().numpy())
                                    group.create_dataset('logvar', data=logvar_avg.cpu().numpy())
                            except Exception as e:
                                print(f"Rank 0 error writing z_file: {e}")
                                raise

                        # Wait for rank 0 to finish writing
                        dist.barrier()
                    else:
                        # In subsequent epochs, read the saved z vectors
                        try:
                            with h5py.File(z_file, 'r') as z_f:
                                z_avg = torch.from_numpy(z_f[file_name]['z'][:]).float().to(self.device_id)
                                mu_avg = torch.from_numpy(z_f[file_name]['mu'][:]).float().to(self.device_id)
                                logvar_avg = torch.from_numpy(z_f[file_name]['logvar'][:]).float().to(self.device_id)
                        except Exception as e:
                            print(f"Rank {self.rank} error reading z_file: {e}")
                            raise

                    dist.barrier()  # Wait for rank 0 to finish writing

                # Training loop
                # Training loop
                self.network.train()
                file_total_loss = 0.0
                file_processed_samples = 0

                for chunk_start in range(0, local_total_points, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, local_total_points)
                    current_batch_size = chunk_end - chunk_start

                    points_chunk = torch.from_numpy(local_points[chunk_start:chunk_end]).float().to(self.device_id)
                    values_chunk = torch.from_numpy(local_values[chunk_start:chunk_end]).float().to(self.device_id)

                    # Synchronize before forward pass
                    dist.barrier()
                    
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    predicted_densities = self.network.module.generator(points_chunk, z_avg)
                    loss = self.combined_loss(predicted_densities, values_chunk)
                    
                    # Synchronize before backward pass for simultaneous backpropagation
                    dist.barrier()
                    loss.backward()
                    
                    # Wait for all GPUs to complete backward pass
                    dist.barrier()
                    
                    # Clip gradients after all GPUs finish backward pass
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                    
                    # Synchronize gradients across GPUs
                    for param in self.network.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data)
                            param.grad.data /= world_size
                    
                    self.optimizer.step()

                    # Calculate weighted loss for this batch
                    loss_value = loss.item() * current_batch_size
                    
                    # All-reduce to get total loss across all processes
                    stats = torch.tensor([loss_value, current_batch_size], device=self.device_id)
                    dist.all_reduce(stats)
                    total_loss = stats[0].item()
                    total_samples = stats[1].item()
                    current_avg_loss = total_loss / total_samples

                    # Update running statistics
                    file_total_loss += total_loss
                    file_processed_samples += total_samples
                    running_avg_loss = file_total_loss / file_processed_samples

                    if self.rank == 0 and (chunk_start // self.chunk_size) % 15 == 0:
                        progress = chunk_start / local_total_points * 100
                        print(f"Progress: {progress:.1f}% - "
                            f"Current Loss: {current_avg_loss:.6f} - "
                            f"Running Avg Loss: {running_avg_loss:.6f}")

                    del points_chunk, values_chunk, predicted_densities, loss
                    torch.cuda.empty_cache()

                stats_tensor = torch.tensor([file_total_loss, file_processed_samples], device=self.device_id)
                dist.all_reduce(stats_tensor)
                
                file_total_loss = stats_tensor[0].item()
                file_total_samples = stats_tensor[1].item()

                epoch_total_loss += file_total_loss
                epoch_total_samples += file_total_samples

                if self.rank == 0:
                    file_avg_loss = file_total_loss / file_total_samples if file_total_samples > 0 else 0
                    print(f"\nFile {file_name} completed - Average Loss: {file_avg_loss:.6f}")

                torch.cuda.synchronize()
                dist.barrier()

            epoch_stats = torch.tensor([epoch_total_loss, epoch_total_samples], device=self.device_id)
            dist.all_reduce(epoch_stats)
            
            epoch_avg_loss = epoch_stats[0].item() / epoch_stats[1].item() if epoch_stats[1].item() > 0 else float('inf')

            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch Summary:")
                print(f"Average Loss: {epoch_avg_loss:.6f}")
                print(f"Time: {epoch_time:.2f} seconds")

            return epoch_avg_loss

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save checkpoint with optional best model saving."""
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': self.network.module.state_dict() if isinstance(self.network, DDP) else self.network.state_dict(),
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