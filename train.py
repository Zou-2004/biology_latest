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
# import pdb


# print("My code GPU count:", torch.cuda.device_count())
# print("My code device count before init:", torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
def _monitor_gpu_usage(self):
    if self.rank == 0:  # Only monitor on primary GPU
        print("\nGPU Usage Statistics:")
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            print(f"GPU {i}:")
            print(f"  Allocated: {memory_allocated/1e9:.2f} GB")
            print(f"  Reserved: {memory_reserved/1e9:.2f} GB")

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
        self.z_dim = 256
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.grad_clip = 5.0
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        
        # Initialize model and move to GPU
        self.network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim).to(self.device_id)
        self.network = DDP(self.network, device_ids=[self.device_id])

        # Create optimizer with explicit initial_lr
        self.optimizer = optim.AdamW([
            {'params': self.network.parameters(), 'initial_lr': config.learning_rate}
        ], lr=config.learning_rate, betas=(config.beta1, 0.999))

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
        
        # Error-based importance weighting
        error_weights = torch.ones_like(point_wise_mse)
        high_error_mask = point_wise_mse > torch.quantile(point_wise_mse, 0.5)
        error_weights[high_error_mask] = 2.0
        
        # Combined weighting
        weights = value_weights * error_weights
        
        # Normalize weights to prevent scaling issues
        weights = weights / weights.mean()
        
        # Calculate weighted loss
        weighted_mse = (point_wise_mse * weights).mean()
        
        return weighted_mse

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
            print(f"Learning rate: {config.learning_rate}")
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
                print("Z vector file not found. Will compute z vectors in first epoch.")
            else:
                # Check if data files have changed
                with h5py.File(data_dir, 'r') as data_f, h5py.File(z_file_path, 'r') as z_f:
                    data_files = set(data_f.keys())
                    z_files = set(z_f.keys())
                    if data_files != z_files:
                        compute_z = True
                        print("Data files have changed. Will recompute z vectors in first epoch.")
                    else:
                        print("Using existing z vectors from file.")

        # Broadcast compute_z decision to all processes
        compute_z = torch.tensor([compute_z], dtype=torch.bool, device=self.device_id)
        dist.broadcast(compute_z, src=0)
        compute_z = compute_z.item()

        best_loss = float('inf')
        start_epoch = 0

        try:
            for epoch in range(start_epoch, config.epoch):
                epoch_start_time = time.time()
                dist.barrier()  # Synchronize before epoch starts
                
                if self.rank == 0:
                    print(f"\nEpoch {epoch + 1}/{config.epoch}")
                    print("=" * 40)

                # Compute z vectors only in first epoch if needed
                use_precomputed_z = not compute_z or epoch > 0
                save_z = compute_z and epoch == 0

                self.network.train()
                epoch_loss = self.train_epoch(
                    epoch, 
                    config, 
                    data_dir, 
                    z_file_path,
                    save_z=save_z,
                    use_precomputed_z=use_precomputed_z
                )
                    
                # Step the learning rate scheduler
                self.scheduler.step()
                
                # Calculate and log training metrics
                if self.rank == 0:
                    epoch_time = time.time() - epoch_start_time
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    print(f"\nEpoch Statistics:")
                    print(f"Time taken: {epoch_time:.2f} seconds")
                    print(f"Average loss: {epoch_loss:.6f}")
                    print(f"Learning rate: {current_lr:.6e}")
                    
                    # Save checkpoints
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        self.save_checkpoint(epoch, epoch_loss, is_best=True)
                        print(f"New best model saved! Loss: {epoch_loss:.6f}")
                    else:
                        self.save_checkpoint(epoch, epoch_loss, is_best=False)
                        print(f"Checkpoint saved. Best loss so far: {best_loss:.6f}")
                    
                    # Log training progress
                    if (epoch + 1) % 10 == 0:
                        print(f"\nTraining Progress: {(epoch + 1)/config.epoch*100:.1f}%")
                        print(f"Best loss so far: {best_loss:.6f}")

                # Synchronize processes before starting next epoch
                dist.barrier()

        except Exception as e:
            print(f"Rank {self.rank} encountered error: {str(e)}")
            raise e

        finally:
            # Cleanup
            try:
                dist.barrier()
                dist.destroy_process_group()
                if self.rank == 0:
                    print("\nTraining completed!")
                    print(f"Best model loss: {best_loss:.6f}")
                    print(f"Model saved at: {self.checkpoint_path}")
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")

    def train_epoch(self, epoch, config, data_file, z_file, save_z=False, use_precomputed_z=True):
        epoch_start_time = time.time()
        samples_processed = 0
        world_size = dist.get_world_size()

        with h5py.File(data_file, 'r') as f:
            file_names = list(f.keys())
            
            # Synchronize file names across processes
            if self.rank == 0:
                np.random.shuffle(file_names)
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

            total_files = len(file_names)
            epoch_total_loss = 0.0
            epoch_total_samples = 0

            # Open z_file in appropriate mode
            z_file_mode = 'a' if save_z and self.rank == 0 else 'r'
            with h5py.File(z_file, z_file_mode) as z_f:
                for file_idx, file_name in enumerate(file_names):
                    if self.rank == 0:
                        print(f"\nProcessing file {file_idx + 1}/{total_files}: {file_name}")
                    
                    # Load and distribute data
                    points = f[file_name]['points'][:]
                    values = f[file_name]['values'][:]
                    total_points = len(points)
                    
                    points_per_gpu = total_points // world_size
                    start_idx = self.rank * points_per_gpu
                    end_idx = start_idx + points_per_gpu if self.rank != world_size - 1 else total_points
                    
                    local_points = points[start_idx:end_idx]
                    local_values = values[start_idx:end_idx]
                    local_total_points = len(local_points)

                    # Handle z vectors
                    if use_precomputed_z and file_name in z_f:
                        z_avg = torch.from_numpy(z_f[file_name]['z'][:]).float().to(self.device_id)
                        mu_avg = torch.from_numpy(z_f[file_name]['mu'][:]).float().to(self.device_id)
                        logvar_avg = torch.from_numpy(z_f[file_name]['logvar'][:]).float().to(self.device_id)
                    else:
                        # Compute z vectors if needed
                        self.network.module.encoder.eval()
                        z_sum = torch.zeros(self.z_dim, device=self.device_id)
                        mu_sum = torch.zeros(self.z_dim, device=self.device_id)
                        logvar_sum = torch.zeros(self.z_dim, device=self.device_id)
                        total_chunks = 0

                        for chunk_start in range(0, local_total_points, self.chunk_size):
                            chunk_end = min(chunk_start + self.chunk_size, local_total_points)
                            points_chunk = torch.from_numpy(local_points[chunk_start:chunk_end]).float().to(self.device_id)
                            
                            with torch.no_grad():
                                mu, logvar, z = self.network.module.encoder(points_chunk)
                                z_sum += z.sum(dim=0)
                                mu_sum += mu.sum(dim=0)
                                logvar_sum += logvar.sum(dim=0)
                                total_chunks += (chunk_end - chunk_start)

                        # Synchronize z vectors across GPUs
                        stats = torch.stack([z_sum, mu_sum, logvar_sum, torch.tensor(total_chunks, device=self.device_id)])
                        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                        
                        total_samples = stats[3].item()
                        z_avg = stats[0] / total_samples
                        mu_avg = stats[1] / total_samples
                        logvar_avg = stats[2] / total_samples

                        if save_z and self.rank == 0:
                            if file_name in z_f:
                                del z_f[file_name]
                            group = z_f.create_group(file_name)
                            group.create_dataset('z', data=z_avg.cpu().numpy())
                            group.create_dataset('mu', data=mu_avg.cpu().numpy())
                            group.create_dataset('logvar', data=logvar_avg.cpu().numpy())

                    # Training loop
                    self.network.train()
                    file_total_loss = 0.0
                    file_processed_samples = 0

                    for chunk_start in range(0, local_total_points, self.chunk_size):
                        chunk_end = min(chunk_start + self.chunk_size, local_total_points)
                        current_batch_size = chunk_end - chunk_start
                        
                        points_chunk = torch.from_numpy(local_points[chunk_start:chunk_end]).float().to(self.device_id)
                        values_chunk = torch.from_numpy(local_values[chunk_start:chunk_end]).float().to(self.device_id)

                        self.optimizer.zero_grad()
                        predicted_densities = self.network.module.generator(points_chunk, z_avg)
                        loss = self.combined_loss(predicted_densities, values_chunk)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                        self.optimizer.step()

                        # Update loss statistics
                        current_loss = loss.item()
                        file_total_loss += current_loss * current_batch_size
                        file_processed_samples += current_batch_size

                        # Progress reporting
                        if self.rank == 0 and (chunk_start // self.chunk_size) % 15 == 0:
                            progress = (chunk_start + self.chunk_size) / local_total_points * 100
                            current_avg = file_total_loss / file_processed_samples if file_processed_samples > 0 else 0
                            print(f"Progress: {progress:.1f}% - "
                                f"Current Loss: {current_loss:.6f} - "
                                f"Running Avg Loss: {current_avg:.6f}")

                        del points_chunk, values_chunk, predicted_densities, loss
                        torch.cuda.empty_cache()

                    # Synchronize file statistics across GPUs
                    file_stats = torch.tensor([file_total_loss, file_processed_samples], 
                                        dtype=torch.float64, device=self.device_id)
                    dist.all_reduce(file_stats, op=dist.ReduceOp.SUM)
                    
                    file_total_loss = file_stats[0].item()
                    file_total_samples = file_stats[1].item()
                    
                    epoch_total_loss += file_total_loss
                    epoch_total_samples += file_total_samples

                    if self.rank == 0:
                        file_avg_loss = file_total_loss / file_total_samples if file_total_samples > 0 else 0
                        print(f"\nFile {file_name} completed - Average Loss: {file_avg_loss:.6f}")

            # Calculate final epoch statistics
            epoch_stats = torch.tensor([epoch_total_loss, epoch_total_samples], 
                                    dtype=torch.float64, device=self.device_id)
            dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)
            
            final_total_loss = epoch_stats[0].item()
            final_total_samples = epoch_stats[1].item()
            
            final_avg_loss = final_total_loss / final_total_samples if final_total_samples > 0 else float('inf')

            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch Summary:")
                print(f"Total Samples: {final_total_samples}")
                print(f"Average Loss: {final_avg_loss:.6f}")
                print(f"Time: {epoch_time:.2f} seconds")

            return final_avg_loss

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
