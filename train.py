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
        self.chunk_size = 100000
        self.z_dim = 256
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.grad_clip = 5.0
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.z_file_path = "/home/zcy/seperate_VAE/checkpoint/fixed_z_vectors.hdf5"

        # Get number of files and initialize fixed z vectors
        with h5py.File(os.path.join(self.data_dir, 'density_data.hdf5'), 'r') as f:
            self.num_files = len(f.keys())
            file_names = list(f.keys())
            if self.rank == 0:
                print(f"Total number of files: {self.num_files}")
                if not os.path.exists(self.z_file_path):
                    self.fixed_file_z = {fname: torch.randn(self.z_dim) for fname in file_names}
                    with h5py.File(self.z_file_path, 'w') as zf:
                        for fname, z in self.fixed_file_z.items():
                            zf.create_dataset(fname, data=z.cpu().numpy())
                    if os.path.exists(self.z_file_path):
                        print("Generated and saved new fixed z vectors to", self.z_file_path)
                        print("Fixed z vectors:", {k: v[:5].tolist() for k, v in self.fixed_file_z.items()})
                    else:
                        raise RuntimeError(f"Failed to save fixed z vectors to {self.z_file_path}")
                else:
                    with h5py.File(self.z_file_path, 'r') as zf:
                        zf_keys = set(zf.keys())
                        missing_keys = [fname for fname in file_names if fname not in zf_keys]
                        if missing_keys:
                            raise KeyError(f"Missing keys in {self.z_file_path}: {missing_keys}")
                        self.fixed_file_z = {fname: torch.tensor(zf[fname][:]) for fname in file_names}
                    print("Loaded fixed z vectors from", self.z_file_path)
                    # print("Fixed z vectors:", {k: v[:5].tolist() for k, v in self.fixed_file_z.items()})
            else:
                self.fixed_file_z = {fname: None for fname in file_names}  # Initialize with None

            # Broadcast fixed_file_z explicitly
            if self.rank == 0:
                broadcast_data = [(fname, z) for fname, z in self.fixed_file_z.items()]
            else:
                broadcast_data = [(fname, None) for fname in file_names]
            dist.barrier()
            dist.broadcast_object_list(broadcast_data, src=0)
            self.fixed_file_z = {fname: z for fname, z in broadcast_data}
            self.current_epoch_z = self.fixed_file_z

        # Initialize network
        self.network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim).to(self.device_id)
        self.network = DDP(
            self.network,
            device_ids=[self.device_id],
            find_unused_parameters=False,
            broadcast_buffers=True
        )

        # Setup optimizer
        world_size = dist.get_world_size()
        self.scaled_lr = config.learning_rate * world_size
        self.optimizer = optim.Adam([
            {'params': self.network.parameters(), 'lr': self.scaled_lr},
        ])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)
        self.checkpoint_name = 'IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

    def combined_loss(self, predicted_density, true_density):
        point_wise_mse = F.mse_loss(predicted_density, true_density, reduction='none')
        value_weights = (true_density.abs() + 1e-4) / (true_density.abs().mean() + 1e-4)
        value_weights = value_weights**2
        error_weights = torch.ones_like(point_wise_mse)
        high_error_mask = point_wise_mse > torch.quantile(point_wise_mse, 0.90)
        error_weights[high_error_mask] = 3.0
        weights = value_weights * error_weights
        weights = weights / weights.mean()
        weighted_mse = (point_wise_mse * weights).mean()
        return weighted_mse
    
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
            total_params = sum(p.numel() for p in self.network.parameters())
            trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            print(f"\nModel Statistics:")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Parameters per GPU: {trainable_params/torch.cuda.device_count():,}")
            print("-" * 40 + "\n")

        data_dir = os.path.join(self.data_dir, 'density_data.hdf5')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data file not found: {data_dir}")
        
        best_loss = float('inf')
        start_epoch = 0

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
                file_z = self.fixed_file_z[file_name].to(self.device_id)  # Fixed random z

                points_per_gpu = total_points // world_size
                remainder = total_points % world_size
                start_idx = self.rank * points_per_gpu + min(self.rank, remainder)
                end_idx = start_idx + points_per_gpu + (1 if self.rank < remainder else 0)
                local_total_points = end_idx - start_idx

                self.network.train()
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

                    predicted_densities = self.network.module.generator(points_chunk, file_z)
                    recon_loss = self.combined_loss(predicted_densities, values_chunk)
                    loss = recon_loss
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                    for param in self.network.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data)
                            param.grad.data /= world_size
                    
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

                    if self.rank == 0 and (chunk_start // self.chunk_size) % 30 == 0:
                        progress = chunk_start / local_total_points * 100
                        print(f"Progress: {progress:.1f}% | Recon Loss: {recon_loss:.6f} | Local Total Loss: {local_loss:.6f}")
                    del points_chunk, values_chunk, predicted_densities, loss
                    torch.cuda.empty_cache()

                if self.rank == 0:
                    self.current_epoch_z[file_name] = file_z.detach().cpu()
                    file_avg_loss = file_total_loss / file_processed_samples
                    print(f"\nFile {file_name} completed: Average Loss: {file_avg_loss:.6f}")

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
            if is_best:
                best_model = {
                    'model_state_dict': self.network.module.state_dict() if isinstance(self.network, DDP) else self.network.state_dict(),
                    'epoch': epoch,
                    'loss': loss,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                torch.save(best_model, os.path.join(self.checkpoint_path, 'model_best.pth'))
                print(f"Saved new best model at epoch {epoch} with loss {loss:.6f}")