# test_logical_two_server.py
import torch
from server import (
    secure_aggregate_client_updates,
    plaintext_sum_client_updates,
    max_abs_diff,
)

def main():
    # 模拟 3 个 client，每个 client 上传 2 层“梯度”
    client_updates = [
        [torch.randn(4, 3), torch.randn(3)],
        [torch.randn(4, 3), torch.randn(3)],
        [torch.randn(4, 3), torch.randn(3)],
    ]

    secure_sum, server0, server1 = secure_aggregate_client_updates(
        client_updates,
        scale=10**6,
    )
    plain_sum = plaintext_sum_client_updates(client_updates)

    err = max_abs_diff(secure_sum, plain_sum)
    print(f"max reconstruction error: {err:.8f}")

    for cid in server0.client_shares:
        print(f"server0 has share of client {cid}")
    for cid in server1.client_shares:
        print(f"server1 has share of client {cid}")

if __name__ == "__main__":
    main()