from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch


# 使用 64-bit 整数环做两方加法秘密共享
# 这里用 2^63，便于后续把环元素映射回有符号整数
RING_MOD = 2 ** 62
DEFAULT_SCALE = 10 ** 7


def _mod_ring(x: torch.Tensor, q: int = RING_MOD) -> torch.Tensor:
    return torch.remainder(x, q)


def _to_fixed_point(x: torch.Tensor, scale: int = DEFAULT_SCALE) -> torch.Tensor:
    """
    float tensor -> int64 tensor
    """
    return torch.round(x * scale).to(torch.int64)


def _from_fixed_point(x_int: torch.Tensor, scale: int = DEFAULT_SCALE) -> torch.Tensor:
    """
    int64 tensor -> float tensor
    """
    return x_int.to(torch.float32) / scale


def _signed_from_ring(x: torch.Tensor, q: int = RING_MOD) -> torch.Tensor:
    """
    把环元素映射回有符号整数区间 [-q/2, q/2)
    """
    half_q = q // 2
    return torch.where(x >= half_q, x - q, x)


def share_tensor_list(
    tensors: List[torch.Tensor],
    scale: int = DEFAULT_SCALE,
    q: int = RING_MOD,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    对一组浮点张量做两方秘密共享：
        x -> (s0, s1), 满足 s0 + s1 = x_int mod q

    返回两组 shares，与输入层结构一致。
    """
    shares0: List[torch.Tensor] = []
    shares1: List[torch.Tensor] = []

    for t in tensors:
        x_int = _to_fixed_point(t.detach().cpu(), scale=scale)

        # 采样 share0
        s0 = torch.randint(
            low=0,
            high=q,
            size=x_int.shape,
            dtype=torch.int64,
        )
        # share1 = x - share0 mod q
        s1 = _mod_ring(x_int - s0, q=q)

        shares0.append(s0)
        shares1.append(s1)


    return shares0, shares1


def reconstruct_tensor_list(
    shares0: List[torch.Tensor],
    shares1: List[torch.Tensor],
    scale: int = DEFAULT_SCALE,
    q: int = RING_MOD,
) -> List[torch.Tensor]:
    """
    从两组 shares 重构出浮点张量列表
    """
    if len(shares0) != len(shares1):
        raise ValueError("shares0 and shares1 must have the same number of layers")

    result: List[torch.Tensor] = []
    for s0, s1 in zip(shares0, shares1):
        x_ring = _mod_ring(s0 + s1, q=q)
        x_signed = _signed_from_ring(x_ring, q=q)
        x_float = _from_fixed_point(x_signed, scale=scale)
        result.append(x_float)

    return result


def add_share_lists(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    q: int = RING_MOD,
) -> List[torch.Tensor]:
    """
    两组 shares 按层相加
    """
    if len(a) != len(b):
        raise ValueError("share lists must have the same number of layers")

    out: List[torch.Tensor] = []
    for x, y in zip(a, b):
        if x.shape != y.shape:
            raise ValueError("share tensor shapes do not match")
        out.append(_mod_ring(x + y, q=q))
    return out


@dataclass
class LogicalServer:
    """
    逻辑服务器：
    - 只保存自己拿到的那一份 share
    - 在本地完成 share 聚合
    """
    server_id: int
    q: int = RING_MOD
    client_shares: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    aggregate_share: Optional[List[torch.Tensor]] = None

    def receive_client_share(self, client_id: int, share: List[torch.Tensor]) -> None:
        if client_id in self.client_shares:
            raise ValueError(f"client_id {client_id} already exists on server {self.server_id}")
        self.client_shares[client_id] = [x.clone() for x in share]

    def aggregate_all_shares(self) -> List[torch.Tensor]:
        if not self.client_shares:
            raise ValueError(f"server {self.server_id} has no client shares")

        agg: Optional[List[torch.Tensor]] = None
        for _, share in self.client_shares.items():
            if agg is None:
                agg = [x.clone() for x in share]
            else:
                agg = add_share_lists(agg, share, q=self.q)

        self.aggregate_share = agg
        return [x.clone() for x in self.aggregate_share]

    def get_aggregate_share(self) -> List[torch.Tensor]:
        if self.aggregate_share is None:
            raise ValueError(f"server {self.server_id} aggregate_share is empty, call aggregate_all_shares() first")
        return [x.clone() for x in self.aggregate_share]

    def clear_round(self) -> None:
        self.client_shares.clear()
        self.aggregate_share = None


def secure_aggregate_client_updates(
    client_updates: List[List[torch.Tensor]],
    scale: int = DEFAULT_SCALE,
    q: int = RING_MOD,
) -> Tuple[List[torch.Tensor], LogicalServer, LogicalServer]:
    """
    一步完成：
    client updates
        -> 两方分享
        -> 两个逻辑服务器分别聚合
        -> 重构出聚合结果

    返回:
        reconstructed_aggregate, server0, server1
    """
    server0 = LogicalServer(server_id=0, q=q)
    server1 = LogicalServer(server_id=1, q=q)

    for client_id, update in enumerate(client_updates):
        share0, share1 = share_tensor_list(update, scale=scale, q=q)
        server0.receive_client_share(client_id, share0)
        server1.receive_client_share(client_id, share1)

    agg_share0 = server0.aggregate_all_shares()
    agg_share1 = server1.aggregate_all_shares()

    reconstructed = reconstruct_tensor_list(
        agg_share0,
        agg_share1,
        scale=scale,
        q=q,
    )
    return reconstructed, server0, server1


def plaintext_sum_client_updates(
    client_updates: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    明文求和，用于对照验证
    """
    if not client_updates:
        raise ValueError("client_updates is empty")

    num_layers = len(client_updates[0])
    result = [torch.zeros_like(client_updates[0][i]).cpu() for i in range(num_layers)]

    for update in client_updates:
        if len(update) != num_layers:
            raise ValueError("inconsistent number of layers in client_updates")
        for i, t in enumerate(update):
            result[i] += t.detach().cpu()

    return result


def max_abs_diff(
    xs: List[torch.Tensor],
    ys: List[torch.Tensor],
) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have same number of layers")

    max_err = 0.0
    for x, y in zip(xs, ys):
        err = (x.detach().cpu() - y.detach().cpu()).abs().max().item()
        max_err = max(max_err, err)
    return max_err