"""Module: src/offline_worker/models/model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1031,
        hidden_channels: int = 256,
        out_channels: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index)
        return F.normalize(z, p=2, dim=-1)  # Chuẩn hóa L2 vector z


class LinkPredictor(nn.Module):
    """Bilinear Link Predictor: score = z_u^T W z_v + bias
    Tránh hiện tượng bão hòa 0.0000 của MLP sâu trên đồ thị thưa.
    """

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_channels, in_channels))
        nn.init.xavier_uniform_(self.W)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        # (batch, d) x (d, d) -> (batch, d)
        transformed_u = torch.matmul(z_u, self.W)
        # Tổng nhân phần tử (dot product) -> (batch,)
        scores = (transformed_u * z_v).sum(dim=-1) + self.bias
        return scores

    def predict_proba(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(z_u, z_v))


class GNNPrerequisiteModel(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 1031,
        sage_hidden: int = 256,
        node_out_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=node_in_dim,
            hidden_channels=sage_hidden,
            out_channels=node_out_dim,
            dropout=dropout,
        )
        self.predictor = LinkPredictor(in_channels=node_out_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_pairs: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(x, edge_index)
        src, dst = edge_pairs[0], edge_pairs[1]
        return self.predictor(z[src], z[dst])