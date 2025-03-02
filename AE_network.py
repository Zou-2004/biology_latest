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
        """
        Args:
            points: Tensor of shape [N, 3], where N can vary
            z: Tensor of shape [1, z_dim] - single z vector for the file
        """
        # Get current batch size (can be different for last chunk)
        num_points = points.size(0)
        
        # Apply positional encoding to points
        points_encoded = positional_encoding(points, self.num_frequencies, include_input=self.include_input)  # [N, encoded_dim]
        
        # Expand z to match current points batch size
        zs = z.expand(num_points, -1)  # [N, z_dim]
        
        # Concatenate encoded points and z
        pointz = torch.cat([points_encoded, zs], dim=-1)  # [N, input_dim]
        
        # Process through network
        l1 = F.leaky_relu(self.linear_1(pointz), negative_slope=0.02)
        l2 = F.leaky_relu(self.linear_2(l1), negative_slope=0.02)
        l3 = F.leaky_relu(self.linear_3(l2), negative_slope=0.02)
        l4 = F.leaky_relu(self.linear_4(l3), negative_slope=0.02)
        l5 = F.leaky_relu(self.linear_5(l4), negative_slope=0.02)
        l6 = F.leaky_relu(self.linear_6(l5), negative_slope=0.02)
        l7 = self.linear_7(l6)  # [N, 1]

        return l7  
    

class encoder(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(encoder, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        
        self.density_net = nn.Sequential(
            nn.Linear(1, self.ef_dim),
            nn.LayerNorm(self.ef_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(self.ef_dim, self.ef_dim*2),
            nn.LayerNorm(self.ef_dim*2),
            nn.LeakyReLU(0.02),
            nn.Linear(self.ef_dim*2, self.ef_dim*4),
            nn.LayerNorm(self.ef_dim*4),
            nn.LeakyReLU(0.02)
        )
        
        self.aggregation_net = nn.Linear(self.ef_dim*4, self.ef_dim*8)
        self.norm_agg = nn.LayerNorm(self.ef_dim*8)
        self.fc_z = nn.Linear(self.ef_dim*8, self.z_dim)
    
    def forward(self, density_values):
        if len(density_values.shape) == 1:
            density_values = density_values.unsqueeze(-1)
        # Chain operations to minimize memory
        z = self.fc_z(
            F.leaky_relu(
                self.norm_agg(self.aggregation_net(self.density_net(density_values))),
                negative_slope=0.02
            )
        ).mean(dim=0)  # [z_dim]
        return z

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
