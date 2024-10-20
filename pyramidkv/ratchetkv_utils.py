import math
import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache


class RatchetKVCluster:
    def __init__(self, window_size=64, num_dims=128, num_orders=2):
        self.window_size = window_size
        self.number_orders = num_orders
        self.num_dims = num_dims

        self.kernel_dim = sum([num_dims ** i for i in range(num_orders + 1)])

        self.mapping_matrix = torch.normal(0, 1 / (self.window_size ** 0.5),
                                           size=(self.kernel_dim, self.window_size))

    def reset(self, window_size=64, num_dims=128, num_orders=2):
        self.window_size = window_size
        self.number_orders = num_orders

        self.num_dims = num_dims

        self.kernel_dim = sum([num_dims ** i for i in range(num_orders + 1)])

        self.mapping_matrix = torch.normal(0, 1 / (self.window_size ** 0.5),
                                           size=(self.kernel_dim, self.window_size))


def init_ratchetkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 64
        if not hasattr(self.config, 'num_orders'):
            self.config.num_orders = 2

    self.kv_cluster = RatchetKVCluster(
        window_size=self.config.window_size,
        num_dims=self.config.hidden_size // self.config.num_attention_heads,
        num_orders=self.config.num_orders
    )


class RatchetKVCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.phi_key_value_cache: List[torch.Tensor] = []
        self.phi_key_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.phi_key_value_cache[layer_idx], self.phi_key_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.phi_key_value_cache[layer_idx], self.phi_key_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.phi_key_value_cache)

    def update(
        self,
        phi_key_value_sum: torch.Tensor,
        phi_key_sum: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            phi_key_value_sum (`torch.Tensor`):
                The new key states to cache.
            phi_key_sum (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens = phi_key_value_sum.shape[-2]

        # Update the cache
        if len(self.phi_key_value_cache) <= layer_idx:
            self.phi_key_value_cache.append(phi_key_value_sum)
            self.phi_key_cache.append(phi_key_sum)
        else:
            self.phi_key_value_cache[layer_idx] = phi_key_value_sum
            self.phi_key_cache[layer_idx] = phi_key_sum

        return self.phi_key_value_cache[layer_idx], self.phi_key_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None


def phi_order2(x, mapping_matrix):
    shape = x.shape[:-1]
    d = x.shape[-1]
    x = x / math.sqrt(math.sqrt(d))
    device = x.device

    zero_order = torch.ones(shape + (1, )).to(device)
    first_order = x

    x = x.unsqueeze(-1)
    second_order = 0.5 * (torch.matmul(x, x.unsqueeze(-1).transpose(-1, -2))).view(*(shape + (d*d, ))).to(device)
    return torch.matmul(torch.cat([zero_order, first_order, second_order], dim=-1), mapping_matrix.to(device))
