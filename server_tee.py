from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch


# ============================================================
# MPC 基础参数与固定点转换
# ============================================================

RING_MOD = 2 ** 62
DEFAULT_SCALE = 10 ** 7


def _mod_ring(x: torch.Tensor, q: int = RING_MOD) -> torch.Tensor:
    return torch.remainder(x, q)


def _to_fixed_point(x: torch.Tensor, scale: int = DEFAULT_SCALE) -> torch.Tensor:
    """
    浮点张量 -> 固定点整数张量。
    """
    return torch.round(x * scale).to(torch.int64)


def _from_fixed_point(x_int: torch.Tensor, scale: int = DEFAULT_SCALE) -> torch.Tensor:
    """
    固定点整数张量 -> 浮点张量。
    """
    return x_int.to(torch.float32) / scale


def _signed_from_ring(x: torch.Tensor, q: int = RING_MOD) -> torch.Tensor:
    """
    将整数环元素恢复为有符号整数区间 [-q/2, q/2)。
    """
    half_q = q // 2
    return torch.where(x >= half_q, x - q, x)


# ============================================================
# 两方加法秘密共享
# ============================================================

def share_tensor_list(
    tensors: List[torch.Tensor],
    scale: int = DEFAULT_SCALE,
    q: int = RING_MOD,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    对一组浮点张量执行两方加法秘密共享：
        x -> (s0, s1), 满足 s0 + s1 = x_int mod q

    返回两组 shares，与输入层结构一致。
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
    """
    从两组 shares 重构出浮点张量列表。
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
    两组 share 按层相加。
    """
    if len(a) != len(b):
        raise ValueError("share lists must have the same number of layers")

    out: List[torch.Tensor] = []

    for x, y in zip(a, b):
        if x.shape != y.shape:
            raise ValueError("share tensor shapes do not match")
        out.append(_mod_ring(x + y, q=q))

    return out


# ============================================================
# TEE 轻量封装
# ============================================================

@dataclass
class TEESealedShare:
    """
    逻辑 TEE 封装后的份额对象。
    """
    sealed: bool
    payload: List[torch.Tensor]


def tee_seal_share(share: List[torch.Tensor]) -> TEESealedShare:
    """
    逻辑 TEE 封装：模拟 MPC 份额进入 TEE 并生成受保护份额。
    """
    return TEESealedShare(
        sealed=True,
        payload=[x.clone() for x in share],
    )


def tee_unseal_share(sealed_share: TEESealedShare) -> List[torch.Tensor]:
    """
    逻辑 TEE 内部解封：只有 TEE 内部处理函数应调用。
    """
    if not sealed_share.sealed:
        raise ValueError("share is not sealed by TEE")

    return [x.clone() for x in sealed_share.payload]


@dataclass
class LogicalTEEServer:
    """
    逻辑 TEE 聚合服务器：
    - 外部只接收 TEE 封装后的份额；
    - 聚合时在 TEE 内部解封并处理；
    - 为兼容原 FL_MPC 的 main.py，提供 client_shares 属性别名。
    """
    server_id: int
    q: int = RING_MOD
    sealed_client_shares: Dict[int, TEESealedShare] = field(default_factory=dict)
    aggregate_share: Optional[List[torch.Tensor]] = None

    @property
    def client_shares(self) -> Dict[int, TEESealedShare]:
        """
        兼容 FL_MPC 中 LogicalServer.client_shares 的日志写法。
        """
        return self.sealed_client_shares

    def receive_sealed_client_share(self, client_id: int, sealed_share: TEESealedShare) -> None:
        if client_id in self.sealed_client_shares:
            raise ValueError(f"client_id {client_id} already exists on server {self.server_id}")

        self.sealed_client_shares[client_id] = sealed_share

    def tee_process_and_aggregate(self) -> List[torch.Tensor]:
        """
        TEE 内部处理：
        1. 解封受保护份额；
        2. 在 TEE 内部执行份额聚合；
        3. 输出聚合份额。
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
            raise ValueError(
                f"server {self.server_id} aggregate_share is empty, call tee_process_and_aggregate() first"
            )

        return [x.clone() for x in self.aggregate_share]

    def clear_round(self) -> None:
        self.sealed_client_shares.clear()
        self.aggregate_share = None


# ============================================================
# FL + MPC + TEE 安全聚合闭环
# ============================================================

def secure_aggregate_client_updates_with_tee(
    client_updates: List[List[torch.Tensor]],
    scale: int = DEFAULT_SCALE,
    q: int = RING_MOD,
) -> Tuple[List[torch.Tensor], LogicalTEEServer, LogicalTEEServer]:
    """
    FL + MPC + TEE 闭环：

        client updates
            -> 固定点量化
            -> 两方加法秘密共享
            -> MPC 份额进入 TEE
            -> TEE 受保护安全份额
            -> TEE 内部解封和份额聚合
            -> 聚合份额重构
            -> 聚合后的全局训练更新参数

    返回：
        reconstructed_aggregate, logical_tee_server0, logical_tee_server1
    """
    server0 = LogicalTEEServer(server_id=0, q=q)
    server1 = LogicalTEEServer(server_id=1, q=q)

    print("[INFO] 参数安全转换模块调用成功。")
    print(f"[INFO] 参数安全转换配置：固定点量化缩放因子={scale}，整数环模数={q}。")

    for client_id, update in enumerate(client_updates):
        share0, share1 = share_tensor_list(update, scale=scale, q=q)

        if client_id == 0:
            print("[INFO] 固定点量化整数训练参数完成。")
            print("[INFO] 整数训练参数映射至整数环完成。")
            print("[INFO] 开始执行两方加法秘密共享处理：随机生成第一份秘密份额，并基于环上减法生成第二份秘密份额。")
            print("[INFO] 两方MPC计算份额生成成功。")

        sealed_share0 = tee_seal_share(share0)
        sealed_share1 = tee_seal_share(share1)

        if client_id == 0:
            print("[INFO] TEE架构可信执行环境适配模块调用成功，MPC计算份额成功转换为TEE侧受保护参数对象。")

        server0.receive_sealed_client_share(client_id, sealed_share0)
        server1.receive_sealed_client_share(client_id, sealed_share1)

    print(
        f"[INFO] TEE安全环境接收完成：Server0接收份额数量={len(server0.sealed_client_shares)}，"
        f"Server1接收份额数量={len(server1.sealed_client_shares)}。"
    )

    agg_share0 = server0.tee_process_and_aggregate()
    agg_share1 = server1.tee_process_and_aggregate()

    print("[INFO] TEE安全环境完成加密份额解封、受保护处理及份额聚合计算。")

    reconstructed = reconstruct_tensor_list(
        agg_share0,
        agg_share1,
        scale=scale,
        q=q,
    )

    print("[INFO] TEE内部聚合结果重构完成，生成聚合后的全局训练更新参数。")

    return reconstructed, server0, server1


def secure_aggregate_client_updates(
    client_updates: List[List[torch.Tensor]],
    scale: int = DEFAULT_SCALE,
    q: int = RING_MOD,
) -> Tuple[List[torch.Tensor], LogicalTEEServer, LogicalTEEServer]:
    """
    映射到 TEE 版本的安全聚合流程。
    """
    return secure_aggregate_client_updates_with_tee(
        client_updates=client_updates,
        scale=scale,
        q=q,
    )


# ============================================================
# 明文对照与误差计算
# ============================================================

def plaintext_sum_client_updates(
    client_updates: List[List[torch.Tensor]],
) -> List[torch.Tensor]:
    """
    明文求和，用于对照验证 TEE + MPC 聚合结果。
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
    """
    计算两组张量列表之间的最大绝对误差。
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have same number of layers")

    max_err = 0.0

    for x, y in zip(xs, ys):
        err = (x.detach().cpu() - y.detach().cpu()).abs().max().item()
        max_err = max(max_err, err)

    return max_err
