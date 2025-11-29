import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math


class Transpose(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class MHGraphAttn(Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, num_heads=4, bias=True, residual=True):
        super(MHGraphAttn, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads * out_features)
        else:
            self.project = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, require_weights=False):
        batch = inputs.shape[0]
        
        # 1. Linear Projection
        support = torch.matmul(inputs, self.weight)
        support = support.reshape(batch, -1, self.num_heads, self.out_features).permute(dims=(0, 2, 1, 3))
        
        # 2. Calculate Raw Scores (e_ij)
        f_1 = torch.matmul(support, self.weight_u).reshape(batch, self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(batch, self.num_heads, -1, 1)
        logits = f_1 + f_2
        weight = self.leaky_relu(logits) 
        
        # 3. [FIX] Strict Masking
        # We need to broadcast adj_mat to [Batch, Heads, N, N]
        # adj_mat is [Batch, N, N] -> unsqueeze head dim -> [Batch, 1, N, N]
        mask = adj_mat.unsqueeze(1)
        
        # Where mask == 0, set weight to -1e9 (Negative Infinity)
        # This ensures Softmax(weight) = 0.0 for these edges.
        # We use masked_fill for safety.
        masked_weight = weight.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax (Normalize)
        attn_weights = torch.softmax(masked_weight, dim=3)
        
        # 5. Aggregate
        support = torch.matmul(attn_weights, support)
        support = support.permute(dims=(0, 2, 1, 3)).reshape(batch, -1, self.num_heads * self.out_features)
        
        if self.bias is not None:
            support = support + self.bias
        if self.residual:
            support = support + self.project(inputs)
            
        support = torch.tanh(support)
        
        if require_weights:
            return support, attn_weights
        else:
            return support, None


class PairNorm(nn.Module):
    def __init__(self, mode="PN", scale=1):
        assert mode in ["None", "PN", "PN-SI", "PN-SCS"]
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == "None":
            return x
        col_mean = x.mean(dim=1, keepdim=True)
        if self.mode == "PN":
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=2).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == "PN-SI":
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=2, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == "PN-SCS":
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=2, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


class HeteFusionAttn(Module):
    def __init__(self, in_features, hidden_size=128, act=nn.Tanh()):
        super(HeteFusionAttn, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            act,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, inputs, require_weights=False):
        # inputs: [batch, num_types, num_nodes, features]
        w = self.project(inputs)
        beta = torch.softmax(w, dim=1)
        fused = (beta * inputs).sum(1)
        if require_weights:
            return fused, beta
        return fused, None


class TemporalHGAT(nn.Module):
    def __init__(self, num_stocks, input_dim=6, lookback=30,
                 hidden_dim=128, num_heads=4, no_ind=False, no_neg=False):
        super(TemporalHGAT, self).__init__()
        self.lookback = lookback
        self.num_stocks = num_stocks
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.no_ind = no_ind
        self.no_neg = no_neg

        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Spatial encoder
        self.ind_gat = MHGraphAttn(hidden_dim, hidden_dim, num_heads=num_heads)
        self.pos_gat = MHGraphAttn(hidden_dim, hidden_dim, num_heads=num_heads)
        self.neg_gat = MHGraphAttn(hidden_dim, hidden_dim, num_heads=num_heads)

        self.ind_mlp = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.pos_mlp = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.neg_mlp = nn.Linear(hidden_dim * num_heads, hidden_dim)

        # Previous weights projection
        self.prev_proj = nn.Linear(1, hidden_dim)

        self.sem_gat = HeteFusionAttn(in_features=hidden_dim, hidden_size=hidden_dim)
        self.pn = PairNorm(mode="PN-SI")

        # Output head per node
        self.output_head = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)

    def forward(self, inputs, require_weights=False):
        # Observation layout: [ind, pos, neg, ts_flat]
        batch = inputs.shape[0]
        adj_size = self.num_stocks * self.num_stocks

        ts_size = self.num_stocks * self.lookback * self.input_dim
        expected_len = 3 * adj_size + ts_size + self.num_stocks  # +prev_weights
        if inputs.shape[1] != expected_len:
            raise ValueError(
                f"TemporalHGAT expected obs length {expected_len} (num_stocks={self.num_stocks}, "
                f"lookback={self.lookback}, input_dim={self.input_dim}, ts_size={ts_size}, "
                f"prev_weights={self.num_stocks}) but got {inputs.shape[1]}"
            )

        ptr = 0
        ind_adj = inputs[:, ptr:ptr + adj_size].reshape(batch, self.num_stocks, self.num_stocks)
        ptr += adj_size
        pos_adj = inputs[:, ptr:ptr + adj_size].reshape(batch, self.num_stocks, self.num_stocks)
        ptr += adj_size
        neg_adj = inputs[:, ptr:ptr + adj_size].reshape(batch, self.num_stocks, self.num_stocks)
        ptr += adj_size

        ts_flat = inputs[:, ptr:ptr + ts_size]
        ptr += ts_size
        prev_weights = inputs[:, ptr:]

        ts_features = ts_flat.reshape(batch * self.num_stocks, self.lookback, self.input_dim)

        # Temporal encoding
        _, (h_n, _) = self.lstm(ts_features)
        node_embeddings = h_n[-1].reshape(batch, self.num_stocks, -1)

        # Spatial encoding
        ind_supp, ind_attn = self.ind_gat(node_embeddings, ind_adj, require_weights)
        pos_supp, pos_attn = self.pos_gat(node_embeddings, pos_adj, require_weights)
        neg_supp, neg_attn = self.neg_gat(node_embeddings, neg_adj, require_weights)

        ind_supp = self.ind_mlp(ind_supp)
        pos_supp = self.pos_mlp(pos_supp)
        neg_supp = self.neg_mlp(neg_supp)

        prev_emb = self.prev_proj(prev_weights.unsqueeze(-1))

        stack_list = [node_embeddings]
        if not self.no_ind:
            stack_list.extend([ind_supp, pos_supp])
        if not self.no_neg:
            stack_list.append(neg_supp)
        stack_list.append(prev_emb)
        stacked = torch.stack(stack_list, dim=2)

        fused_embedding, sem_attn = self.sem_gat(stacked.permute(0, 2, 1, 3), require_weights)
        fused_embedding = self.pn(fused_embedding)

        scores = self.output_head(fused_embedding).squeeze(-1)

        if require_weights:
            attn_payload = {
                "industry": ind_attn,
                "positive": pos_attn,
                "negative": neg_attn,
                "semantic": sem_attn,
            }
            return scores, attn_payload

        return scores
