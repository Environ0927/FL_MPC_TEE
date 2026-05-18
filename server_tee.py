from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch


RING_MOD = 2 ** 62
DEFAULT_SCALE = 10 ** 7


def _mod_ring(x: torch.Tensor, q: int = RING_MOD) -> torch.Tensor:
    return torch.remainder(x, q)


def _to_fixed_point(x: torch.Tensor, scale: int = DEFAULT_SCALE) -> torch.Tensor:
    return torch.round(x * scale).to(torch.int64)


def _from_fixed_point(x_int: torch.Tensor, scale: int = DEFAULT_SCALE) -> torch.Tensor:
    return x_int.to(torch.float32) / scale


def _signed_from_ring(x: torch.Tensor, q: int = RING_MOD) -> torch.Tensor:
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
    """
    shares0: List[torch.Tensor] = []
    shares1: List[torch.Tensor] = []

    for t in tensors:
        x_int = _to_fixed_point(t.detach().cpu(), scale=scale)
        s0 = torch.randint(
            low=0,
            high=int(q),
            size=x_int.shape,
            dtype=torch.int64,
        )
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
    if len(a) != len(b):
        raise ValueError("share lists must have the same number of layers")

    out: List[torch.Tensor] = []
    for x, y in zip(a, b):
        if x.shape != y.shape:
            raise ValueError("share tensor shapes do not match")
        out.append(_mod_ring(x + y, q=q))
    return out


# =========================
# TEE 轻量封装
# =========================

@dataclass
class TEESealedShare:
    """
    逻辑TEE封装后的份额对象。
    外部只看到 sealed 标记，不直接处理里面的 share。
    """
    sealed: bool
    payload: List[torch.Tensor]


def tee_seal_share(share: List[torch.Tensor]) -> TEESealedShare:
    """
    逻辑TEE封装：模拟份额进入TEE保护边界。
    这里不做真实加密，只做接口隔离。
    """
    return TEESealedShare(
        sealed=True,
        payload=[x.clone() for x in share],
    )


def tee_unseal_share(sealed_share: TEESealedShare) -> List[torch.Tensor]:
    """
    逻辑TEE内部解封：只有TEE函数内部应调用此接口。
    """
    if not sealed_share.sealed:
        raise ValueError("share is not sealed by TEE")
    return [x.clone() for x in sealed_share.payload]


@dataclass
class LogicalTEEServer:
    """
    逻辑TEE服务器：
    - 外部只接收“TEE封装后的份额”
    - 聚合时只能在 TEE 函数内部解封并处理
    """
    server_id: int
    q: int = RING_MOD
    sealed_client_shares: Dict[int, TEESealedShare] = field(default_factory=dict)
    aggregate_share: Optional[List[torch.Tensor]] = None

    def receive_sealed_client_share(self, client_id: int, sealed_share: TEESealedShare) -> None:
        if client_id in self.sealed_client_shares:
            raise ValueError(f"client_id {client_id} already exists on server {self.server_id}")
        self.sealed_client_shares[client_id] = sealed_share

    def tee_process_and_aggregate(self) -> List[torch.Tensor]:
        """
        TEE封装函数：
        只有在这里才把份额取出并参与聚合。
        """
        if not self.sealed_client_shares:
            raise ValueError(f"server {self.server_id} has no sealed client shares")

        agg: Optional[List[torch.Tensor]] = None
        for _, sealed_share in self.sealed_client_shares.items():
            share = tee_unseal_share(sealed_share)
            if agg is None:
                agg = [x.clone() for x in share]
            else:
                agg = add_share_lists(agg, share, q=self.q)

        self.aggregate_share = agg
        return [x.clone() for x in self.aggregate_share]

    def get_aggregate_share(self) -> List[torch.Tensor]:
        if self.aggregate_share is None:
            raise ValueError(f"server {self.server_id} aggregate_share is empty, call tee_process_and_aggregate() first")
        return [x.clone() for x in self.aggregate_share]

    def clear_round(self) -> None:
        self.sealed_client_shares.clear()
        self.aggregate_share = None


def secure_aggregate_client_updates_with_tee(
    client_updates: List[List[torch.Tensor]],
    scale: int = DEFAULT_SCALE,
    q: int = RING_MOD,
) -> Tuple[List[torch.Tensor], LogicalTEEServer, LogicalTEEServer]:
    """
    FL + MPC + TEE(逻辑封装)闭环：
    client updates
        -> 两方秘密共享
        -> 份额进入 TEE 封装
        -> 两个逻辑TEE服务器接收封装份额
        -> TEE内部处理并聚合
        -> 重构出聚合结果
    """
    server0 = LogicalTEEServer(server_id=0, q=q)
    server1 = LogicalTEEServer(server_id=1, q=q)

    for client_id, update in enumerate(client_updates):
        share0, share1 = share_tensor_list(update, scale=scale, q=q)

        sealed_share0 = tee_seal_share(share0)
        sealed_share1 = tee_seal_share(share1)

        server0.receive_sealed_client_share(client_id, sealed_share0)
        server1.receive_sealed_client_share(client_id, sealed_share1)

    agg_share0 = server0.tee_process_and_aggregate()
    agg_share1 = server1.tee_process_and_aggregate()

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