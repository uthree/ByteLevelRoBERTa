import torch
import torch.nn as nn

class SeparableConv1d(nn.Module):
    def __init__(self, dim, kernel_size=5, activation=nn.GELU):
        super().__init__()
        self.spatial = nn.Conv1d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
        self.depthwise = nn.Conv1d(dim, dim, 1, 1, 0)
        self.activation = activation()

    def forward(self, x):
        res = x
        x = self.spatial(x)
        x = self.depthwise(x)
        x = x + res
        return x


class ByteEmbedding(nn.Module):
    def __init__(self, num_special_token=3, embedding_dim=512, byte_dim=128, concat_kernel=4, conv_kernel=5, conv_depth=1, activation=nn.GELU):
        super().__init__()
        self.embedding = nn.Embedding(256 + num_special_token, byte_dim)
        self.conv = nn.Sequential(*[SeparableConv1d(byte_dim, byte_dim, activation=activation) for _ in range(conv_depth)])
        self.last_conv = nn.Conv1d(byte_dim, embedding_dim, concat_kernel, concat_kernel, 0)

    def forward(self, x): # x: [Batch, Length]
        x = self.embedding(x) # x [Batch, byte_dim, Length]
        x = torch.transpose(x, 1, 2) # x [Batch, Length byte_dim]
        x = self.conv(x)
        x = self.last_conv(x)
        x = torch.transpose(x, 1, 2) # x [Batch, embedding_dim, Length]
        return x

class ByteUnembedding(nn.Module):
    def __init__(self, num_special_token=3, embedding_dim=512, concat_kernel=4):
        super().__init__()
        self.conv = nn.ConvTranspose1d(embedding_dim, 256+num_special_token, concat_kernel, 1, 0)

    def forward(self, x): # x: [Batch, embedding_dim, Length]
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = torch.transpose(x, 1, 2)
        return x

    
