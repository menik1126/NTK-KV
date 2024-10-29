import math
import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache


# class RatchetKVCluster:
#     def __init__(self, window_size=64, num_dims=128):
#         self.window_size = window_size
#         self.number_orders = num_orders
#         self.num_dims = num_dims
#
#         self.kernel_dim = sum([num_dims ** i for i in range(num_orders + 1)])
#
#         self.mapping_matrix = torch.normal(0, 1 / (self.window_size ** 0.5),
#                                            size=(self.kernel_dim, self.window_size))
#
#     def reset(self, window_size=64, num_dims=128, num_orders=2):
#         self.window_size = window_size
#         self.number_orders = num_orders
#
#         self.num_dims = num_dims
#
#         self.kernel_dim = sum([num_dims ** i for i in range(num_orders + 1)])
#
#         self.mapping_matrix = torch.normal(0, 1 / (self.window_size ** 0.5),
#                                            size=(self.kernel_dim, self.window_size))
#
#     def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
#
#         # check if prefix phase
#         assert key_states.shape[-2] == query_states.shape[-2]
#         bsz, num_heads, q_len, head_dim = query_states.shape
#
#         print(f"StreamingLLM max_capacity_prompt {self.max_capacity_prompt}")
#
#         if q_len < self.max_capacity_prompt:
#             return key_states, value_states
#         else:
#             # attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
#             # mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
#             # mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
#             # mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
#             # mask = mask.to(attn_weights.device)
#             # attention_mask = mask[None, None, :, :]
#
#             # attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
#
#             # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#             # attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
#             # if self.pooling == 'avgpool':
#             #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
#             # elif self.pooling == 'maxpool':
#             #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
#             # else:
#             #     raise ValueError('Pooling method not supported')
#             # attn_cache = attn_weights_sum
#             # indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
#
#             indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(
#                 key_states.device)
#             indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)
#
#             k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
#             v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
#             k_cur = key_states[:, :, -self.window_size:, :]
#             v_cur = value_states[:, :, -self.window_size:, :]
#             key_states = torch.cat([k_past_compress, k_cur], dim=2)
#             value_states = torch.cat([v_past_compress, v_cur], dim=2)
#             return key_states, value_states


def init_ratchetkv(self):
    if not hasattr(self, "init_ratchet"):
        self.init_ratchet = True
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 2048
        if not hasattr(self.config, 'max_capacity'):
            self.config.max_capacity = 4
        if not hasattr(self.config, 'safe_limit'):
            self.config.safe_limit = 2

        self.config.window_size -= 2 * self.head_dim + self.config.max_capacity + 1

        self.phi_key_value_sum = 0
        self.phi_key_sum = 0

        self.phi = TaylorFeature(self.head_dim)


def _safe_scale_attn(self, attn_output, lse, query_states, phi_key_value_sum, phi_key_sum):
    phi_query_states = self.phi(query_states)

    se = lse.exp().unsqueeze(-1)
    new_se = se + (phi_query_states * phi_key_sum).sum(-1, keepdim=True)

    ret = attn_output * (se / new_se).transpose(1, 2) + (
                torch.matmul(phi_query_states, phi_key_value_sum) / new_se).transpose(1, 2)
    ret = ret.where((ret / attn_output - 1).abs() < (self.config.safe_limit - 1), attn_output)
    # print(torch.norm(ret - attn_output))
    return ret


class TaylorFeature:
    def __init__(self, head_dim):
        self.d = head_dim
        self.rd = math.sqrt(head_dim)
        self.rrd = math.sqrt(self.rd)
        self.r2 = math.sqrt(2)

    def elu(self, x):
        ret = x
        ret[ret < 0] = ret[ret < 0].exp()
        return ret

    def __call__(self, x):
        x = x / self.rrd

        return torch.cat([
            torch.ones_like(x[..., 0:1]),
            self.elu(x),
            (x ** 2) / self.r2,
            # z[..., self.ids_non_diag[0], self.ids_non_diag[1]] / math.sqrt(2)
        ], dim=-1)



# def phi_order2(x):
#     d = x.shape[-1]
#     x = x / math.sqrt(math.sqrt(d))
#
#     return torch.cat([
#         torch.ones_like(x[..., 0:1]),
#         x,
#         x ** 2 / (2 ** 0.5)
#     ], dim=-1)

    # z = torch.einsum("...i,...j->...ij", x, x)
    #
    # ids_non_diag = torch.triu_indices(d, d, 1)
    # ids_diag = torch.arange(0, d)
    #
    # if mapping_matrix is not None:
    #     return torch.matmul(
    #         torch.cat([
    #             torch.ones_like(x[..., 0:1]),
    #             x,
    #             z[..., ids_diag, ids_diag] / 2,
    #             z[..., ids_non_diag[0], ids_non_diag[1]] / math.sqrt(2)
    #         ], dim=-1),
    #         mapping_matrix
    #     )
    # else:
    #     return torch.cat([
    #         torch.ones_like(x[..., 0:1]),
    #         x,
    #         z[..., ids_diag, ids_diag] / 2,
    #         z[..., ids_non_diag[0], ids_non_diag[1]] / math.sqrt(2)
    #     ], dim=-1)

    # ret = 0
    # ret = ret + torch.matmul(torch.ones(shape + (1,)).to(device), mapping_matrix[0:1].to(device))
    # ret = ret + torch.matmul(x, mapping_matrix[1:1 + d].to(device))
    # x = x.unsqueeze(-1)
    # ret = ret + torch.matmul(0.5 * (torch.matmul(x, x.transpose(-1, -2))).view(*(shape + (d * d,))).to(device),
    #                          mapping_matrix[1 + d:1 + d + d ** 2].to(device))
    # return ret
    # # zero_order = torch.ones(shape + (1, )).to(device)
    # first_order = x
    #
    # x = x.unsqueeze(-1)
    # second_order = 0.5 * (torch.matmul(x, x.transpose(-1, -2))).view(*(shape + (d*d, ))).to(device)
    # return torch.matmul(torch.cat([zero_order, first_order, second_order], dim=-1), mapping_matrix.to(device))
