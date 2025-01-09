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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class generator(nn.Module):
    def __init__(self, z_dim, point_dim, gf_dim):
        super(generator, self).__init__()
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.gf_dim = gf_dim
        
        self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
        self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
        self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
        self.linear_7 = nn.Linear(self.gf_dim*1, 1, bias=True)
        
        # Initialize weights
        for layer in [self.linear_1, self.linear_2, self.linear_3, 
                     self.linear_4, self.linear_5, self.linear_6]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.constant_(layer.bias, 0)
        
        nn.init.normal_(self.linear_7.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_7.bias, 0)

    def forward(self, points, z, chunk_size):
        

        # Ensure consistent shapes
        if len(points.shape) == 2:
            points = points.unsqueeze(0)  # [1, N, 3]
        
        batch_size = points.size(0)
        num_points = points.size(1)
        
        # Ensure z has correct shape
        if len(z.shape) == 1:
            z = z.unsqueeze(0)  # [1, z_dim]
        elif len(z.shape) == 2 and z.size(0) != batch_size:
            z = z.repeat(batch_size, 1)
        
        # Expand z to match points
        zs = z.unsqueeze(1).expand(batch_size, num_points, -1)
        
        # Concatenate points and z
        pointz = torch.cat([points, zs], dim=-1)
        
        # Process in chunks consistent with the chunk_size below
        
        outputs = []
        
        for i in range(0, num_points, chunk_size):
            end_idx = min(i + chunk_size, num_points)
            chunk = pointz[:, i:end_idx, :]
            
            # Maintain batch dimension
            chunk_flat = chunk.reshape(-1, self.z_dim + self.point_dim)
            
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

class IM_SVAE(object):
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_to_keep = 1  # Add this line
        self.point_dim = 3
        self.ef_dim = 256
        self.gf_dim = 256
        self.z_dim = 8
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.grad_clip = 5.0
        self.chunk_size = 100000
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        # Create separate networks for positive and negative values
        self.pos_network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)
        self.neg_network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)
        
        self.pos_network.to(self.device)
        self.neg_network.to(self.device)
        
        initial_lr = config.learning_rate
        
        # Separate optimizers for positive and negative networks
        self.pos_optimizer = optim.Adam([
            {'params': self.pos_network.parameters(), 'initial_lr': initial_lr}
        ], lr=initial_lr, betas=(config.beta1, 0.999))
        
        self.neg_optimizer = optim.Adam([
            {'params': self.neg_network.parameters(), 'initial_lr': initial_lr}
        ], lr=initial_lr, betas=(config.beta1, 0.999))

        # Separate schedulers
        self.pos_scheduler = optim.lr_scheduler.StepLR(
            self.pos_optimizer, step_size=30, gamma=0.1,
            last_epoch=config.epoch-1 if hasattr(config, 'epoch') else -1
        )
        
        self.neg_scheduler = optim.lr_scheduler.StepLR(
            self.neg_optimizer, step_size=30, gamma=0.1,
            last_epoch=config.epoch-1 if hasattr(config, 'epoch') else -1
        )

        # Separate checkpoints
        self.pos_checkpoint_name = 'IM_VAE_positive.model'
        self.neg_checkpoint_name = 'IM_VAE_negative.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.beta = 0.01  # KL divergence weight


    def combined_loss(self, predicted_density, true_density, mu, logvar):
        # Add shape validation
        assert predicted_density.shape == true_density.shape, \
            f"Shape mismatch in density tensors: predicted {predicted_density.shape} vs true {true_density.shape}"
        assert mu.shape == logvar.shape, \
            f"Shape mismatch in latent tensors: mu {mu.shape} vs logvar {logvar.shape}"
        
        # Reconstruction loss
        mse = self.mse_loss(predicted_density, true_density)
        l1 = self.l1_loss(predicted_density, true_density)
        recon_loss = 0.7 * mse + 0.3 * l1
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss
    

    @property
    def model_dir(self):
        return "{}_density_svae".format(self.dataset_name)
    
    def train(self, config):
        # Check input files
        pos_train_file = os.path.join(self.data_dir, 'positive_density_data.hdf5')
        neg_train_file = os.path.join(self.data_dir, 'negative_density_data.hdf5')
        
        if not os.path.exists(pos_train_file):
            raise FileNotFoundError(f"Positive data file not found: {pos_train_file}")
        if not os.path.exists(neg_train_file):
            raise FileNotFoundError(f"Negative data file not found: {neg_train_file}")

        # Print data summary
        with h5py.File(pos_train_file, 'r') as f:
            pos_files = list(f.keys())
            print(f"\nPositive data files: {len(pos_files)}")
            for key in pos_files:
                print(f"  {key}: {f[key]['points'].shape[0]} points")
        
        with h5py.File(neg_train_file, 'r') as f:
            neg_files = list(f.keys())
            print(f"\nNegative data files: {len(neg_files)}")
            for key in neg_files:
                print(f"  {key}: {f[key]['points'].shape[0]} points")

        best_pos_loss = float('inf')
        best_neg_loss = float('inf')
        
        print(f"\n----------Training Summary----------")
        print(f"Z vector dimension: {self.z_dim}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Total epochs: {config.epoch}")
        print(f"Learning rate: {config.learning_rate}")
        print("-" * 40 + "\n")

        for epoch in range(config.epoch):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{config.epoch}")
            print("=" * 40)

            # Train positive network
            pos_loss = self.train_network(
                epoch, config, pos_train_file, 
                self.pos_network, self.pos_optimizer,
                "positive"
            )

            # Train negative network
            neg_loss = self.train_network(
                epoch, config, neg_train_file,
                self.neg_network, self.neg_optimizer,
                "negative"
            )

            # Step schedulers
            self.pos_scheduler.step()
            self.neg_scheduler.step()

            # Save best models and z-vectors
            if pos_loss < best_pos_loss:
                best_pos_loss = pos_loss
                self.save_network(epoch, "positive")
                self.save_z_vectors(pos_train_file, self.pos_network, "positive")
                print(f"\nNew best positive model! Loss: {pos_loss:.6f}")

            if neg_loss < best_neg_loss:
                best_neg_loss = neg_loss
                self.save_network(epoch, "negative")
                self.save_z_vectors(neg_train_file, self.neg_network, "negative")
                print(f"New best negative model! Loss: {neg_loss:.6f}")

            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch completed in {epoch_time:.2f}s")
            print(f"Current best - Positive: {best_pos_loss:.6f}, Negative: {best_neg_loss:.6f}")

    def train_network(self, epoch, config, train_file, network, optimizer, mode):
        """
        Train network for one epoch.
        
        Args:
            epoch: Current epoch number
            config: Training configuration
            train_file: Path to training data file
            network: Network to train (positive or negative)
            optimizer: Optimizer for the network
            mode: "positive" or "negative"
        
        Returns:
            float: Average loss for the epoch
        """
        try:
            # Initialize training
            torch.cuda.empty_cache()  # Clear GPU memory before training
            network.train()
            epoch_loss_sum = 0
            total_batches = 0
            start_time = time.time()
            
            with h5py.File(train_file, 'r') as f:
                file_names = list(f.keys())
                np.random.shuffle(file_names)  # Shuffle files for each epoch
                total_files = len(file_names)
                print(f"\n{mode.capitalize()} Network - Processing {total_files} files")

                for file_idx, file_name in enumerate(file_names):
                    try:
                        file_start_time = time.time()
                        print(f"\nFile {file_idx+1}/{total_files}: {file_name}")
                        
                        # Load file data
                        points = f[file_name]['points'][:]
                        values = f[file_name]['values'][:]
                        total_points = points.shape[0]
                        
                        # Initialize file statistics
                        file_loss_sum = 0
                        file_batches = 0
                        
                        # Process data in chunks
                        for chunk_start in range(0, total_points, self.chunk_size):
                            try:
                                chunk_end = min(chunk_start + self.chunk_size, total_points)
                                chunk_size = chunk_end - chunk_start
                                
                                # Load and prepare data chunks
                                points_chunk = torch.from_numpy(
                                    points[chunk_start:chunk_end]
                                ).float().to(self.device)
                                
                                values_chunk = torch.from_numpy(
                                    values[chunk_start:chunk_end]
                                ).float().to(self.device)
                                
                                # Forward pass
                                optimizer.zero_grad()
                                mu, logvar, z = network.encoder(points_chunk)
                                predicted_densities = network.generator(points_chunk, z, chunk_size)
                                
                                # Reshape tensors for loss calculation
                                # Add shape checks
                                predicted_densities = predicted_densities.contiguous().view(-1, 1)
                                values_chunk = values_chunk.contiguous().view(-1, 1)
                                assert predicted_densities.shape == values_chunk.shape, f"Shape mismatch: predicted {predicted_densities.shape} vs true {values_chunk.shape}"
                                
                                # Calculate loss
                                loss = self.combined_loss(predicted_densities, values_chunk, mu, logvar)
                                
                                # Backward pass and optimization
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(network.parameters(), self.grad_clip)
                                optimizer.step()
                                
                                # Update statistics
                                current_loss = loss.item()
                                epoch_loss_sum += current_loss
                                total_batches += 1
                                file_loss_sum += current_loss
                                file_batches += 1
                                
                                # Print progress
                                if (chunk_start // self.chunk_size) % 5 == 0:
                                    progress = (chunk_start + chunk_size) / total_points * 100
                                    avg_loss = file_loss_sum / file_batches
                                    print(f"Progress: {progress:.1f}% "
                                        f"[{chunk_start//self.chunk_size + 1}/"
                                        f"{(total_points + self.chunk_size - 1)//self.chunk_size}] "
                                        f"Current Loss: {current_loss:.6f} "
                                        f"Avg Loss: {avg_loss:.6f}")
                                
                                # Memory management
                                del points_chunk, values_chunk, predicted_densities, loss, mu, logvar, z
                                torch.cuda.empty_cache()
                                
                            except Exception as e:
                                print(f"Error processing chunk {chunk_start//self.chunk_size + 1}: {str(e)}")
                                continue
                        
                        # Print file summary
                        if file_batches > 0:
                            file_avg_loss = file_loss_sum / file_batches
                            file_time = time.time() - file_start_time
                            print(f"\nFile Summary - {file_name}")
                            print(f"Average Loss: {file_avg_loss:.6f}")
                            print(f"Processing Time: {file_time:.2f}s")
                            print(f"Points Processed: {total_points}")
                            print(f"Batches Completed: {file_batches}")
                        
                    except Exception as e:
                        print(f"Error processing file {file_name}: {str(e)}")
                        continue
                    
            # Calculate and return average epoch loss
            if total_batches == 0:
                print(f"Warning: No batches processed in {mode} network training")
                return float('inf')
            
            avg_loss = epoch_loss_sum / total_batches
            total_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\n{mode.capitalize()} Network - Epoch {epoch+1} Summary")
            print(f"Average Loss: {avg_loss:.6f}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Total Batches: {total_batches}")
            
            return avg_loss
        
        except Exception as e:
            print(f"Fatal error in {mode} network training: {str(e)}")
            return float('inf')
            
        finally:
            # Ensure memory is cleared
            torch.cuda.empty_cache()
    
    def save_z_vectors(self, data_file, network, mode):
        z_vector_path = f'/home/zcy/seperate_VAE/z_vector_for_VAE_{mode}.hdf5'
        
        try:
            if os.path.exists(z_vector_path):
                os.remove(z_vector_path)
            
            network.eval()
            print(f"\nSaving {mode} z-vectors...")
            
            with h5py.File(data_file, 'r') as f_in, h5py.File(z_vector_path, 'w') as f_out:
                total_files = len(f_in.keys())
                
                for idx, file_name in enumerate(f_in.keys(), 1):
                    try:
                        print(f"Processing {file_name} ({idx}/{total_files})")
                        points = torch.from_numpy(f_in[file_name]['points'][:]).float().to(self.device)
                        
                        # Process in chunks if needed
                        z_vectors = []
                        mu_vectors = []
                        logvar_vectors = []
                        
                        with torch.no_grad():
                            for chunk_start in range(0, points.shape[0], self.chunk_size):
                                chunk_end = min(chunk_start + self.chunk_size, points.shape[0])
                                points_chunk = points[chunk_start:chunk_end]
                                
                                mu, logvar, z = network.encoder(points_chunk)
                                
                                z_vectors.append(z.cpu().numpy())
                                mu_vectors.append(mu.cpu().numpy())
                                logvar_vectors.append(logvar.cpu().numpy())
                        
                        # Combine chunks
                        z_combined = np.concatenate(z_vectors, axis=0)
                        mu_combined = np.concatenate(mu_vectors, axis=0)
                        logvar_combined = np.concatenate(logvar_vectors, axis=0)
                        
                        # Save to file
                        group = f_out.create_group(file_name)
                        group.create_dataset('z', data=z_combined, compression='gzip')
                        group.create_dataset('mu', data=mu_combined, compression='gzip')
                        group.create_dataset('logvar', data=logvar_combined, compression='gzip')
                        
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
                        continue
                    
            print(f"Z-vectors saved to {z_vector_path}")
            
        except Exception as e:
            print(f"Error saving z-vectors: {e}")


    def save_network(self, epoch, mode):
        checkpoint_name = self.pos_checkpoint_name if mode == "positive" else self.neg_checkpoint_name
        network = self.pos_network if mode == "positive" else self.neg_network
        optimizer = self.pos_optimizer if mode == "positive" else self.neg_optimizer
        scheduler = self.pos_scheduler if mode == "positive" else self.neg_scheduler
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # Generate checkpoint path
        save_path = os.path.join(self.checkpoint_path, f"{checkpoint_name}-{epoch}.pth")
        
        # Save the checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, save_path)
        
        # Manage checkpoint files
        checkpoint_prefix = os.path.join(self.checkpoint_path, f"{checkpoint_name}-")
        existing_checkpoints = []
        
        # Get list of existing checkpoints
        for filename in os.listdir(self.checkpoint_path):
            if filename.startswith(checkpoint_name) and filename.endswith(".pth"):
                checkpoint_path = os.path.join(self.checkpoint_path, filename)
                epoch_num = int(filename.split('-')[-1].split('.')[0])
                existing_checkpoints.append((epoch_num, checkpoint_path))
        
        # Sort checkpoints by epoch number
        existing_checkpoints.sort(reverse=True)
        
        # Keep only the most recent checkpoint and delete others
        for i, (epoch_num, checkpoint_path) in enumerate(existing_checkpoints):
            if i >= self.max_to_keep:  # Keep only max_to_keep checkpoints
                try:
                    os.remove(checkpoint_path)
                    print(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    print(f"Error removing checkpoint {checkpoint_path}: {e}")
        
        print(f"Saved {mode} checkpoint: {save_path}")