import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import math

#positional encoding frequency
num_frequencies=10


def positional_encoding(points, num_frequencies, include_input=True, log_sampling=True):
    """
    Apply positional encoding to 3D coordinates (NeRF-style).

    Args:
        points: Tensor of shape [N, 3], N is the number of points, and 3 is (x, y, z).
        num_frequencies: Number of frequency bands for encoding.
        include_input: Whether to include the original coordinates in the output.
        log_sampling: Whether to use logarithmic frequency sampling (default: True).

    Returns:
        Tensor of shape [N, 3 * (2 * num_frequencies) + (3 if include_input else 0)].
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


# class generator(nn.Module):
#     def __init__(self, z_dim, point_dim, gf_dim, num_frequencies, include_input=True):
#         super(generator, self).__init__()
#         self.z_dim = z_dim
#         self.point_dim = point_dim
#         self.gf_dim = gf_dim
#         self.num_frequencies = num_frequencies
#         self.include_input = include_input

#         # Update the input dimension based on positional encoding
#         self.encoded_dim = self.point_dim * (2 * self.num_frequencies)
#         if self.include_input:
#             self.encoded_dim += self.point_dim

#         self.input_dim = self.z_dim + self.encoded_dim

#         # More efficient network structure
#         self.linear_1 = nn.Linear(self.input_dim, self.gf_dim * 4)
#         self.linear_2 = nn.Linear(self.gf_dim * 4, self.gf_dim * 4)
#         self.linear_3 = nn.Linear(self.gf_dim * 4, self.gf_dim * 4)
#         self.linear_4 = nn.Linear(self.gf_dim * 4, self.gf_dim * 2)
#         self.linear_5 = nn.Linear(self.gf_dim * 2, self.gf_dim)
#         self.linear_6 = nn.Linear(self.gf_dim, self.gf_dim)
#         self.linear_7 = nn.Linear(self.gf_dim, 1)

#         #Batch nrmalization
#         self.bn1 = nn.BatchNorm1d(self.gf_dim * 4)
#         self.bn2 = nn.BatchNorm1d(self.gf_dim * 4)
#         self.bn3 = nn.BatchNorm1d(self.gf_dim * 4)
#         self.bn4 = nn.BatchNorm1d(self.gf_dim * 2)
#         self.bn5 = nn.BatchNorm1d(self.gf_dim)
#         self.bn6 = nn.BatchNorm1d(self.gf_dim)

#         # Lighter dropout
#         self.dropout = nn.Dropout(0.05)

#         # Initialize weights
#         for layer in [self.linear_1, self.linear_2, self.linear_3,
#                      self.linear_4, self.linear_5, self.linear_6, 
#                      self.linear_7]:
#             nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
#             nn.init.constant_(layer.bias, 0)

#     def forward(self, points, z):
#         num_points = points.size(0)

#         # Positional encoding
#         points_encoded = positional_encoding(points, self.num_frequencies, include_input=self.include_input)

#         if len(z.shape) == 1:
#             z = z.unsqueeze(0)
#         zs = z.expand(num_points, -1)

#         # Concatenate encoded points and z
#         pointz = torch.cat([points_encoded, zs], dim=-1)

#         # Forward pass with residual connections
#         x1 = self.dropout(F.leaky_relu(self.bn1(self.linear_1(pointz)), 0.2))
        
#         x2 = self.dropout(F.leaky_relu(self.bn2(self.linear_2(x1)), 0.2))
#         x2 = x2 + x1  # Residual connection
        
#         x3 = self.dropout(F.leaky_relu(self.bn3(self.linear_3(x2)), 0.2))
#         x3 = x3 + x2  # Residual connection
        
#         x4 = self.dropout(F.leaky_relu(self.bn4(self.linear_4(x3)), 0.2))
#         x5 = self.dropout(F.leaky_relu(self.bn5(self.linear_5(x4)), 0.2))
#         x6 = self.dropout(F.leaky_relu(self.bn6(self.linear_6(x5)), 0.2))
        
#         # Final layer with density activation
#         density = self.linear_7(x6)

#         return density
    

class generator(nn.Module):
    def __init__(self, z_dim, point_dim, gf_dim, num_frequencies, include_input=True):
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

    def forward(self, points, z):
        # Input points should be [chunk_size, 3]
        # No need to add batch dimension
        
        num_points = points.size(0)  # This is your chunk size

        # Apply positional encoding to points (modify positional_encoding to work without batch dim)
        points_encoded = positional_encoding(points, self.num_frequencies, include_input=self.include_input)  # [chunk_size, encoded_dim]
        # print(points_encoded.shape)
        # Ensure z has the correct shape for concatenation
        if len(z.shape) == 1:
            z = z.unsqueeze(0)  # [1, z_dim]

        # Expand z to match points
        zs = z.expand(num_points, -1)  # [chunk_size, z_dim]

        # Concatenate encoded points and z
        pointz = torch.cat([points_encoded, zs], dim=-1)  # [chunk_size, input_dim]

        # No need to flatten as we don't have batch dimension
        # Pass through the network
        l1 = F.leaky_relu(self.linear_1(pointz), negative_slope=0.02)
        l2 = F.leaky_relu(self.linear_2(l1), negative_slope=0.02)
        l3 = F.leaky_relu(self.linear_3(l2), negative_slope=0.02)
        l4 = F.leaky_relu(self.linear_4(l3), negative_slope=0.02)
        l5 = F.leaky_relu(self.linear_5(l4), negative_slope=0.02)
        l6 = F.leaky_relu(self.linear_6(l5), negative_slope=0.02)
        l7 = self.linear_7(l6)  # This should output [chunk_size, 1]

        return l7  
    

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
        self.num_frequencies=num_frequencies
        self.encoder = encoder(self.ef_dim, self.z_dim)
        self.generator = generator(self.z_dim, self.point_dim, self.gf_dim,self.num_frequencies)
