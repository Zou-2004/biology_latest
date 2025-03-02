import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import numpy as np
import h5py
from AE_network import *

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
        self.ef_dim = 256
        self.gf_dim = 256
        self.chunk_size = 100000//2
        self.z_dim = 256
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.grad_clip = 5.0
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        # self.beta = 0.01  # KL loss weight

        # Get number of files first
        with h5py.File(os.path.join(self.data_dir, 'density_data.hdf5'), 'r') as f:
            self.num_files = len(f.keys())
            if self.rank == 0:
                print(f"Total number of files: {self.num_files}")
        self.current_epoch_z = {}
        # Initialize model and move to GPU
        self.network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim).to(self.device_id)
        self.network = DDP(
            self.network, 
            device_ids=[self.device_id],
            find_unused_parameters=False, 
            broadcast_buffers=True 
        )

        # Setup optimizer with different learning rates
        world_size = dist.get_world_size()
        self.scaled_lr = config.learning_rate * world_size
        self.optimizer = optim.Adam([
            {'params': self.network.parameters(), 'lr': self.scaled_lr},
        ])

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=0.99 
        )

        # Checkpoint configuration
        self.checkpoint_name = 'IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

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
    
    def compute_z_vectors(self, values_chunk):
        with torch.no_grad():
            # Make sure values are in the right shape [N, 1]
            if len(values_chunk.shape) == 1:
                values_chunk = values_chunk.unsqueeze(-1)
                
            z = self.network.module.encoder(values_chunk, is_training=False)
            return z
            
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

        
        best_loss = float('inf')
        start_epoch = 0

        try:
            for epoch in range(start_epoch, config.epoch):
                compute_z = (epoch == 0)
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
                    # compute_z=compute_z
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

    def train_epoch(self, epoch, config, data_file, z_file):
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

                points_per_gpu = total_points // world_size
                remainder = total_points % world_size
                start_idx = self.rank * points_per_gpu + min(self.rank, remainder)
                end_idx = start_idx + points_per_gpu + (1 if self.rank < remainder else 0)
                local_total_points = end_idx - start_idx

                self.network.train()
                file_z = None
                total_weight = 0.0
                file_total_loss = 0.0
                file_processed_samples = 0

                for chunk_start in range(0, local_total_points, self.chunk_size):
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    chunk_end = min(chunk_start + self.chunk_size, local_total_points)
                    current_batch_size = chunk_end - chunk_start

                    points_chunk = torch.from_numpy(points_data[start_idx + chunk_start:start_idx + chunk_end]).float().to(self.device_id)
                    values_chunk = torch.from_numpy(values_data[start_idx + chunk_start:start_idx + chunk_end]).float().to(self.device_id)
                    if len(values_chunk.shape) == 1:
                        values_chunk = values_chunk.unsqueeze(-1)

                    with torch.no_grad():
                        chunk_z = self.network.module.encoder(values_chunk)  # [32], no gradients
                        weight = torch.mean(torch.abs(values_chunk))
                        if file_z is None:
                            file_z = chunk_z * weight
                        else:
                            file_z += chunk_z * weight
                        total_weight += weight
                        current_file_z = file_z / total_weight if total_weight > 0 else chunk_z.clone()

                    predicted_densities = self.network.module.generator(points_chunk, current_file_z)
                    recon_loss = self.combined_loss(predicted_densities, values_chunk)
                    # chunk_z_grad = self.network.module.encoder(values_chunk)  # [32], with gradients
                    # z_reg_loss = torch.mean(chunk_z_grad.pow(2)) * 0.0001  # Regularize encoder output
                    # z_consistency_loss = F.mse_loss(chunk_z_grad, current_file_z)
                    # loss = recon_loss + z_reg_loss + 0.5 * z_consistency_loss
                    loss=recon_loss
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)

                    for param in self.network.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data)
                            param.grad.data /= world_size
                    

                    self.optimizer.step()

                    with torch.no_grad():
                        local_loss = loss.item()  # Local loss for this GPU
                        loss_value = local_loss * current_batch_size
                        stats = torch.tensor([loss_value, current_batch_size], device=self.device_id)
                        dist.all_reduce(stats)
                        chunk_total_loss = stats[0].item()  # Sum of loss * batch_size across GPUs
                        chunk_total_samples = stats[1].item()  # Sum of batch sizes across GPUs
                        
                        file_total_loss += chunk_total_loss
                        file_processed_samples += chunk_total_samples
                        # global_avg_loss = chunk_total_loss / chunk_total_samples  # Global average loss per sample

                    if self.rank == 0 and (chunk_start // self.chunk_size) % 30 == 0:
                        progress = chunk_start / local_total_points * 100
                        print(f"Progress: {progress:.1f}% | Recon Loss: {recon_loss:.6f} | Local Total Loss: {local_loss:.6f}")
                        # print(f"Progress: {progress:.1f}% | "
                        #     f"Recon Loss: {recon_loss:.6f} | "
                        #     f"Z Reg Loss: {z_reg_loss:.6f} | "
                        #     f"Z Consistency Loss: {z_consistency_loss:.6f} | "
                        #     f"Local Total Loss: {local_loss:.6f} | "
                        #     )
                    del points_chunk, values_chunk, predicted_densities, chunk_z, current_file_z, loss
                    torch.cuda.empty_cache()

                if total_weight > 0:
                    file_z = file_z / total_weight
                dist.all_reduce(file_z)
                file_z = file_z / dist.get_world_size()
                if self.rank == 0:
                    self.current_epoch_z[file_name] = file_z.detach().cpu()

                if self.rank == 0:
                    file_avg_loss = file_total_loss / file_processed_samples
                    print(f"\nFile {file_name} completed:")
                    print(f"Average Loss: {file_avg_loss:.6f}")

                epoch_total_loss += file_total_loss
                epoch_total_samples += file_processed_samples

                del file_z
                torch.cuda.empty_cache()

            epoch_stats = torch.tensor([epoch_total_loss, epoch_total_samples], device=self.device_id)
            dist.all_reduce(epoch_stats)
            
            epoch_avg_loss = epoch_stats[0].item() / epoch_stats[1].item()

            if self.rank == 0:
                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch Summary:")
                print(f"Average Loss: {epoch_avg_loss:.6f}")
                print(f"Total Time: {epoch_time:.2f} seconds")

            return epoch_avg_loss

    def save_checkpoint(self, epoch, loss, is_best=False):
        if self.rank == 0:
            os.makedirs(self.checkpoint_path, exist_ok=True)
            
            if is_best:
                # Save the model state
                best_model = {
                    'model_state_dict': self.network.module.state_dict() if isinstance(self.network, DDP) else self.network.state_dict(),
                    'epoch': epoch,
                    'loss': loss,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(best_model, os.path.join(self.checkpoint_path, 'model_best.pth'))
                print(f"Saved new best model at epoch {epoch} with loss {loss:.6f}")
                
                # Save all z vectors for this epoch
                best_z_path = os.path.join(self.checkpoint_path, "best_z_vectors.hdf5")
                with h5py.File(best_z_path, 'w') as f:  # 'w' mode to overwrite previous best
                    for fname, z in self.current_epoch_z.items():
                        f.create_dataset(fname, data=z.numpy())
                print("Saved z vectors for best model")
