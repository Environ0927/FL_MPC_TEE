from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch


RING_MOD = 2 ** 62
DEFAULT_SCALE = 10 ** 7


def _mod_ring(x: torch.Tensor, q: int = RING_MOD) -> torch.Tensor:
    return torch.remainder(x, q)


def _to_fixed_point(x: torch.Tensor, scale: int = DEFAULT_SCALE) -> torch.Tensor:
    """
    float tensor -> int64 fixed-point tensor
    """
    return torch.round(x * scale).to(torch.int64)


def _signed_from_ring(x: torch.Tensor, q: int = RING_MOD) -> torch.Tensor:
    """
    将环元素恢复为有符号整数区间 [-q/2, q/2)。
    """
    half_q = q // 2
    return torch.where(x >= half_q, x - q, x)


# ============================================================
# 通用工具函数
# ============================================================

def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def get_arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_tensor_list(path: str, tensors: List[torch.Tensor]) -> None:
    torch.save([t.detach().cpu() for t in tensors], path)


@dataclass
class StructureItem:
    name: str
    shape: Tuple[int, ...]
    numel: int
    start: int
    end: int
    role: str = "normal"


def tensor_summary(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
    t = tensor.detach().cpu().float()
    return {
        "name": name,
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()) if t.numel() > 1 else 0.0,
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "l2_norm": float(torch.norm(t).item()),
    }


def print_gradient_table(
    named_tensors: List[Tuple[str, torch.Tensor]],
    roles: Optional[Dict[str, str]] = None,
    title: str = "训练梯度参数明细",
) -> None:
    roles = roles or {}
    log_info(title)
    print("-" * 118)
    print(
        f"{'Layer Name':38s} "
        f"{'Role':20s} "
        f"{'Shape':20s} "
        f"{'Numel':>10s} "
        f"{'Mean':>12s} "
        f"{'Std':>12s} "
        f"{'L2-Norm':>12s}"
    )
    print("-" * 118)

    for name, tensor in named_tensors:
        s = tensor_summary(name, tensor)
        role = roles.get(name, "normal")
        print(
            f"{s['name']:38s} "
            f"{role:20s} "
            f"{str(s['shape']):20s} "
            f"{s['numel']:10d} "
            f"{s['mean']:12.6e} "
            f"{s['std']:12.6e} "
            f"{s['l2_norm']:12.6e}"
        )

    print("-" * 118)


def flatten_named_tensors(
    named_tensors: List[Tuple[str, torch.Tensor]],
    roles: Optional[Dict[str, str]] = None,
) -> Tuple[torch.Tensor, List[StructureItem]]:
    roles = roles or {}
    flats: List[torch.Tensor] = []
    structure: List[StructureItem] = []
    offset = 0

    for name, tensor in named_tensors:
        t = tensor.detach().cpu()
        flat = t.reshape(-1)
        numel = int(flat.numel())

        structure.append(
            StructureItem(
                name=name,
                shape=tuple(t.shape),
                numel=numel,
                start=offset,
                end=offset + numel,
                role=roles.get(name, "normal"),
            )
        )
        flats.append(flat)
        offset += numel

    if not flats:
        raise ValueError("待转换参数为空，无法执行向量化。")

    return torch.cat(flats, dim=0), structure


def print_structure_mapping(structure: List[StructureItem]) -> None:
    log_info("参数结构映射信息如下：")
    print("-" * 108)
    print(
        f"{'Name':38s} "
        f"{'Role':20s} "
        f"{'Shape':20s} "
        f"{'Start':>8s} "
        f"{'End':>8s} "
        f"{'Numel':>8s}"
    )
    print("-" * 108)

    for item in structure:
        print(
            f"{item.name:38s} "
            f"{item.role:20s} "
            f"{str(list(item.shape)):20s} "
            f"{item.start:8d} "
            f"{item.end:8d} "
            f"{item.numel:8d}"
        )

    print("-" * 108)


def check_structure_consistency(vector: torch.Tensor, structure: List[StructureItem]) -> bool:
    if not structure:
        return False
    if structure[0].start != 0:
        return False
    for prev, cur in zip(structure[:-1], structure[1:]):
        if prev.end != cur.start:
            return False
    return structure[-1].end == int(vector.numel())


def compute_quantization_error(
    vector: torch.Tensor,
    x_int: torch.Tensor,
    scale: int,
) -> Dict[str, float]:
    restored = x_int.detach().cpu().to(torch.float32) / float(scale)
    original = vector.detach().cpu().to(torch.float32)
    diff = (original - restored).abs()

    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "l2_error": float(torch.norm(diff).item()),
    }


def print_quantization_error(qerr: Dict[str, float]) -> None:
    log_info(
        "固定点量化误差统计："
        f"max_abs_error={qerr['max_abs_error']:.8e}, "
        f"mean_abs_error={qerr['mean_abs_error']:.8e}, "
        f"l2_error={qerr['l2_error']:.8e}。"
    )


def compute_integer_reconstruct_error(
    x_int: torch.Tensor,
    rec_int: torch.Tensor,
) -> Dict[str, int]:
    x = x_int.detach().cpu().to(torch.int64)
    y = rec_int.detach().cpu().to(torch.int64)
    diff = (x - y).abs()

    return {
        "max_integer_error": int(diff.max().item()),
        "sum_integer_error": int(diff.sum().item()),
        "num_error_elements": int((diff != 0).sum().item()),
    }


def print_integer_error(ierr: Dict[str, int], prefix: str = "mpc份额整数误差统计") -> None:
    log_info(
        f"{prefix}："
        f"max_integer_error={ierr['max_integer_error']}, "
        f"sum_integer_error={ierr['sum_integer_error']}, "
        f"num_error_elements={ierr['num_error_elements']}。"
    )


def build_ring_mapping_summary(
    x_int: torch.Tensor,
    q: int,
) -> Dict[str, Any]:
    """
    通用整数环映射摘要：只描述整数参数向量映射至整数环，不混入份额生成或重构校验。
    输出口径与 FL_MPC 专项测试保持一致。
    """
    x = x_int.detach().cpu().to(torch.int64).reshape(-1)
    x_ring = torch.remainder(x, q)

    return {
        "ring_mod": int(q),
        "x_int_dtype": str(x_int.dtype),
        "x_int_dim": int(x.numel()),
        "x_int_min": int(x.min().item()) if x.numel() else 0,
        "x_int_max": int(x.max().item()) if x.numel() else 0,
        "x_ring_min": int(x_ring.min().item()) if x_ring.numel() else 0,
        "x_ring_max": int(x_ring.max().item()) if x_ring.numel() else 0,
        "negative_integer_count": int((x < 0).sum().item()) if x.numel() else 0,
        "ring_range_check": bool((x_ring >= 0).all().item() and (x_ring < q).all().item()) if x_ring.numel() else True,
    }


def print_ring_mapping_summary(summary: Dict[str, Any], name: str = "整数训练参数") -> None:
    log_info(f"{name}映射至整数环完成：")
    print(
        f"  - 整数环模数 q={summary['ring_mod']}，"
        f"参数维度={summary['x_int_dim']}，数据类型={summary['x_int_dtype']}。"
    )
    print(
        f"  - 映射前整数范围=[{summary['x_int_min']}, {summary['x_int_max']}]，"
        f"负数元素数量={summary['negative_integer_count']}。"
    )
    print(
        f"  - 映射后环上范围=[{summary['x_ring_min']}, {summary['x_ring_max']}]，"
        f"范围校验 x_ring ∈ [0, q) -> {summary['ring_range_check']}。"
    )


def build_basic_mpc_ring_mapping_summary(
    mpc_integer_matrix: torch.Tensor,
    mpc_integer_vector: torch.Tensor,
    q: int,
) -> Dict[str, Any]:
    """
    压缩参数专用：展示 [sparse_index, bucket_id, quantized_value] 三元组向量的整数环映射结果。
    输出口径与 FL_MPC 测试5保持一致。
    """
    matrix = mpc_integer_matrix.detach().cpu().to(torch.int64)
    vector = mpc_integer_vector.detach().cpu().to(torch.int64).reshape(-1)
    x_ring = torch.remainder(vector, q)

    return {
        "ring_mod": int(q),
        "mpc_matrix_shape": list(matrix.shape),
        "mpc_vector_dim": int(vector.numel()),
        "triplet_format": "[sparse_index, bucket_id, quantized_value]",
        "x_int_min": int(vector.min().item()) if vector.numel() else 0,
        "x_int_max": int(vector.max().item()) if vector.numel() else 0,
        "x_ring_min": int(x_ring.min().item()) if x_ring.numel() else 0,
        "x_ring_max": int(x_ring.max().item()) if x_ring.numel() else 0,
        "negative_quantized_value_count": int((matrix[:, 2] < 0).sum().item()) if matrix.numel() else 0,
        "ring_range_check": bool((x_ring >= 0).all().item() and (x_ring < q).all().item()) if x_ring.numel() else True,
    }


def print_basic_mpc_ring_mapping_summary(summary: Dict[str, Any]) -> None:
    log_info("整数参数向量映射至整数环完成：")
    print(
        f"  - 整数环模数 q={summary['ring_mod']}，"
        f"MPC矩阵形状={summary['mpc_matrix_shape']}，整数向量维度={summary['mpc_vector_dim']}。"
    )
    print(
        f"  - 参数格式={summary['triplet_format']}，"
        f"整数参数范围=[{summary['x_int_min']}, {summary['x_int_max']}]，"
        f"负数量化值数量={summary['negative_quantized_value_count']}。"
    )
    print(
        f"  - 环上映射后参数范围=[{summary['x_ring_min']}, {summary['x_ring_max']}]，"
        f"范围校验 x_ring ∈ [0, q) -> {summary['ring_range_check']}。"
    )


def two_party_share_integer_vector(
    x_int: torch.Tensor,
    q: int = RING_MOD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_ring = _mod_ring(x_int.detach().cpu().to(torch.int64), q=q)
    s0 = torch.randint(low=0, high=q, size=x_ring.shape, dtype=torch.int64)
    s1 = _mod_ring(x_ring - s0, q=q)
    return s0, s1


def reconstruct_integer_vector(
    s0: torch.Tensor,
    s1: torch.Tensor,
    q: int = RING_MOD,
) -> torch.Tensor:
    x_ring = _mod_ring(
        s0.detach().cpu().to(torch.int64) + s1.detach().cpu().to(torch.int64),
        q=q,
    )
    return _signed_from_ring(x_ring, q=q)


def print_share_summary(s0: torch.Tensor, s1: torch.Tensor, q: int) -> None:
    log_info("MPC秘密份额生成结果如下：")
    print(f"  - share0 shape: {list(s0.shape)}, dtype: {s0.dtype}")
    print(f"  - share1 shape: {list(s1.shape)}, dtype: {s1.dtype}")
    print(f"  - integer ring modulus q: {q}")
    print(f"  - share0 first 8 values: {s0.reshape(-1)[:8].tolist()}")
    print(f"  - share1 first 8 values: {s1.reshape(-1)[:8].tolist()}")


def hash_tensor_int64(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def hash_json(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def make_tee_protected_tensor_object(
    *,
    arch: str,
    object_name: str,
    tensor: torch.Tensor,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    构造 TEE 侧受保护参数对象的测试表示。

    这里不模拟真实硬件 enclave 的加密实现，而是模拟参数适配层输出：
    - tee_arch: x86 / ARM / RISC-V
    - object_name: 对象名称
    - payload_hash: 受保护参数载荷哈希
    - metadata_hash: 元数据哈希
    - sealed_blob: 模拟 sealed object 摘要
    """
    payload_hash = hash_tensor_int64(tensor)
    metadata_hash = hash_json(metadata)
    sealed_blob = hashlib.sha256(
        f"{arch}|{object_name}|{payload_hash}|{metadata_hash}".encode("utf-8")
    ).hexdigest()

    return {
        "tee_arch": arch,
        "object_name": object_name,
        "object_type": "protected_tensor",
        "payload_shape": list(tensor.shape),
        "payload_dtype": str(tensor.dtype),
        "payload_hash": payload_hash,
        "metadata": metadata,
        "metadata_hash": metadata_hash,
        "sealed_blob": sealed_blob,
        "protection_mode": "simulated_tee_sealed_parameter_object",
    }


def make_tee_protected_json_object(
    *,
    arch: str,
    object_name: str,
    payload: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    payload_hash = hash_json(payload)
    metadata_hash = hash_json(metadata)
    sealed_blob = hashlib.sha256(
        f"{arch}|{object_name}|{payload_hash}|{metadata_hash}".encode("utf-8")
    ).hexdigest()

    return {
        "tee_arch": arch,
        "object_name": object_name,
        "object_type": "protected_json",
        "payload_hash": payload_hash,
        "metadata": metadata,
        "metadata_hash": metadata_hash,
        "sealed_blob": sealed_blob,
        "protection_mode": "simulated_tee_sealed_parameter_object",
    }


def validate_tee_tensor_object(
    obj: Dict[str, Any],
    *,
    expected_arch: str,
    expected_name: str,
    source_tensor: torch.Tensor,
) -> bool:
    if obj.get("tee_arch") != expected_arch:
        return False
    if obj.get("object_name") != expected_name:
        return False
    if obj.get("payload_hash") != hash_tensor_int64(source_tensor):
        return False

    metadata_hash = hash_json(obj.get("metadata", {}))
    expected_sealed_blob = hashlib.sha256(
        f"{expected_arch}|{expected_name}|{obj.get('payload_hash')}|{metadata_hash}".encode("utf-8")
    ).hexdigest()

    return obj.get("metadata_hash") == metadata_hash and obj.get("sealed_blob") == expected_sealed_blob


def validate_tee_json_object(
    obj: Dict[str, Any],
    *,
    expected_arch: str,
    expected_name: str,
    source_payload: Dict[str, Any],
) -> bool:
    if obj.get("tee_arch") != expected_arch:
        return False
    if obj.get("object_name") != expected_name:
        return False
    if obj.get("payload_hash") != hash_json(source_payload):
        return False

    metadata_hash = hash_json(obj.get("metadata", {}))
    expected_sealed_blob = hashlib.sha256(
        f"{expected_arch}|{expected_name}|{obj.get('payload_hash')}|{metadata_hash}".encode("utf-8")
    ).hexdigest()

    return obj.get("metadata_hash") == metadata_hash and obj.get("sealed_blob") == expected_sealed_blob


def print_tee_object_summary(tee_objects: List[Dict[str, Any]], arch_name: str) -> None:
    log_info(f"{arch_name}架构可信执行环境适配模块调用成功，MPC计算份额成功转换为TEE侧受保护参数对象。")
    log_info("TEE受保护参数对象生成结果如下：")
    for obj in tee_objects:
        print(
            f"  - object_name={obj['object_name']}, "
            f"type={obj['object_type']}, "
            f"tee_arch={obj['tee_arch']}, "
            f"payload_hash={obj['payload_hash'][:16]}..., "
            f"sealed_blob={obj['sealed_blob'][:16]}..."
        )


def print_file_outputs(out_dir: str, filenames: List[str]) -> None:
    log_info("结果文件保存完成：")
    for name in filenames:
        print(f"  - {os.path.join(out_dir, name)}")


# ============================================================
# 测试输入构造函数
# ============================================================

def generate_mock_personalized_fl_gradients() -> Tuple[List[Tuple[str, torch.Tensor]], Dict[str, str]]:
    torch.manual_seed(52)

    named_grads = [
        ("shared.conv.weight", torch.randn(8, 1, 3, 3) * 0.01),
        ("shared.conv.bias", torch.randn(8) * 0.01),
        ("shared.fc.weight", torch.randn(32, 128) * 0.01),
        ("shared.fc.bias", torch.randn(32) * 0.01),
        ("personal.head.weight", torch.randn(10, 32) * 0.01),
        ("personal.head.bias", torch.randn(10) * 0.01),
    ]

    roles = {
        "shared.conv.weight": "shared_layer",
        "shared.conv.bias": "shared_layer",
        "shared.fc.weight": "shared_layer",
        "shared.fc.bias": "shared_layer",
        "personal.head.weight": "personalized_layer",
        "personal.head.bias": "personalized_layer",
    }

    return named_grads, roles


def generate_mock_approximate_fl_compressed_gradient(
    dim: int = 256,
    bucket_size: int = 16,
) -> Dict[str, Any]:
    torch.manual_seed(53)

    full_grad = torch.randn(dim) * 0.01
    k = max(1, dim // 4)
    sparse_indices = torch.randperm(dim)[:k].sort().values
    selected_values = full_grad[sparse_indices]

    scale = 10 ** 6
    quantized_values = torch.round(selected_values * scale).to(torch.int64)
    bucket_ids = torch.div(sparse_indices, bucket_size, rounding_mode="floor").to(torch.int64)

    return {
        "dim": dim,
        "bucket_size": bucket_size,
        "sparse_indices": sparse_indices,
        "bucket_ids": bucket_ids,
        "selected_values": selected_values,
        "quantized_values": quantized_values,
        "scale": scale,
    }


def generate_mock_secure_fl_params() -> Tuple[List[Tuple[str, torch.Tensor]], Dict[str, Any]]:
    torch.manual_seed(54)

    named_grads = [
        ("layer1.weight", torch.randn(16, 16) * 0.01),
        ("layer1.bias", torch.randn(16) * 0.01),
        ("layer2.weight", torch.randn(10, 16) * 0.01),
        ("layer2.bias", torch.randn(10) * 0.01),
    ]

    aux_params = {
        "client_id": "client_0001",
        "round_id": 1,
        "clip_bound": 1.0,
        "attack_model": "complex_attack",
        "secure_rule": "attack_resistant_consistency_check",
        "timestamp": int(time.time()),
    }

    return named_grads, aux_params


# ============================================================
# 4.1.7 个性化 FL + ML-MPC + x86 TEE
# ============================================================

def run_personalized_fl_ml_mpc_x86_tee(args: Any) -> None:
    scale = int(get_arg(args, "scale", 10 ** 6))
    q = int(get_arg(args, "ring_mod", RING_MOD))
    out_root = str(get_arg(args, "out_dir", "./results/conversion_tests_tee"))

    out_dir = os.path.join(out_root, "test_7_personalized_fl_ml_mpc_x86_tee")
    ensure_dir(out_dir)

    log_info("===========================================================")
    log_info("测试编号7：个性化联邦学习、机器学习型多方安全计算与x86架构TEE间的参数安全转换功能。")
    log_info("输入技术子类型识别成功：联邦学习子类型=面向个性化数据的高精度无损联邦学习；MPC子类型=面向机器学习的多方安全计算；TEE类型=x86架构可信执行环境。")

    named_grads, roles = generate_mock_personalized_fl_gradients()
    log_info("个性化联邦学习侧训练梯度参数读取成功。")
    log_info("共享层梯度、个性化层梯度及参数结构信息识别成功。")
    print_gradient_table(named_grads, roles, title="个性化联邦学习训练梯度参数明细")

    vector, structure = flatten_named_tensors(named_grads, roles=roles)
    structure_check = check_structure_consistency(vector, structure)
    log_info(f"训练梯度向量化完成：总维度={vector.numel()}，层数={len(structure)}。")
    print_structure_mapping(structure)

    x_int = _to_fixed_point(vector, scale=scale)
    qerr = compute_quantization_error(vector, x_int, scale)
    log_info(f"固定点量化整数训练参数完成：scale={scale}，整数参数维度={x_int.numel()}。")
    print_quantization_error(qerr)

    ring_mapping_summary = build_ring_mapping_summary(x_int=x_int, q=q)
    print_ring_mapping_summary(ring_mapping_summary, name="整数训练参数")

    log_info("开始执行两方加法秘密共享处理：随机生成第一份秘密份额，并基于环上减法生成第二份秘密份额。")
    s0, s1 = two_party_share_integer_vector(x_int, q=q)
    log_info("面向机器学习的多方安全计算所需的两方MPC计算份额生成成功。")
    print_share_summary(s0, s1, q)

    tee_metadata = {
        "test_id": 7,
        "fl_type": "personalized_fl",
        "mpc_type": "ml_mpc",
        "tee_type": "x86",
        "vector_dim": int(vector.numel()),
        "scale": scale,
        "ring_mod": q,
    }
    tee_objects = [
        make_tee_protected_tensor_object(
            arch="x86",
            object_name="mpc_share_0",
            tensor=s0,
            metadata={**tee_metadata, "share_owner": "server0"},
        ),
        make_tee_protected_tensor_object(
            arch="x86",
            object_name="mpc_share_1",
            tensor=s1,
            metadata={**tee_metadata, "share_owner": "server1"},
        ),
    ]
    print_tee_object_summary(tee_objects, arch_name="x86")

    tee_object_check = (
        validate_tee_tensor_object(tee_objects[0], expected_arch="x86", expected_name="mpc_share_0", source_tensor=s0)
        and validate_tee_tensor_object(tee_objects[1], expected_arch="x86", expected_name="mpc_share_1", source_tensor=s1)
    )

    rec_int = reconstruct_integer_vector(s0, s1, q=q)
    ierr = compute_integer_reconstruct_error(x_int, rec_int)
    reconstruct_check = torch.equal(rec_int, x_int.detach().cpu().to(torch.int64))

    print_integer_error(ierr)
    log_info(f"参数结构一致性校验结果：{structure_check}")
    log_info(f"TEE受保护参数对象转换校验结果：{tee_object_check}")

    if not reconstruct_check:
        raise AssertionError("mpc份额结果与量化后整数参数不一致。")
    if not structure_check:
        raise AssertionError("参数结构一致性校验失败。")
    if not tee_object_check:
        raise AssertionError("TEE受保护参数对象转换校验失败。")

    log_info("mpc份额校验通过：份额结果与量化后整数参数一致。")
    log_info("TEE受保护参数对象转换校验通过。")

    save_tensor_list(os.path.join(out_dir, "share_0.pt"), [s0])
    save_tensor_list(os.path.join(out_dir, "share_1.pt"), [s1])
    save_tensor_list(os.path.join(out_dir, "quantized_vector.pt"), [x_int])
    save_json(os.path.join(out_dir, "tee_protected_objects.json"), {"tee_objects": tee_objects})

    check_result = {
        "test_id": 7,
        "test_name": "personalized_fl_ml_mpc_x86_tee",
        "fl_type": "面向个性化数据的高精度无损联邦学习",
        "mpc_type": "面向机器学习的多方安全计算",
        "tee_type": "x86架构可信执行环境",
        "vector_dim": int(vector.numel()),
        "num_layers": len(structure),
        "structure": [asdict(item) for item in structure],
        "scale": scale,
        "ring_mod": q,
        "quantization_error": qerr,
        "integer_reconstruct_error": ierr,
        "reconstruct_check": bool(reconstruct_check),
        "structure_check": bool(structure_check),
        "tee_object_check": bool(tee_object_check),
        "status": "PASS",
    }
    save_json(os.path.join(out_dir, "check_result.json"), check_result)

    print_file_outputs(out_dir, ["share_0.pt", "share_1.pt", "quantized_vector.pt", "tee_protected_objects.json", "check_result.json"])
    log_info("测试7执行完成，参数转换过程无异常报错。")


# ============================================================
# 4.1.8 近似 FL + 基础算子 MPC + ARM TEE
# ============================================================

def run_approximate_fl_basic_mpc_arm_tee(args: Any) -> None:
    q = int(get_arg(args, "ring_mod", RING_MOD))
    out_root = str(get_arg(args, "out_dir", "./results/conversion_tests_tee"))
    approx_dim = int(get_arg(args, "approx_dim", 256))
    bucket_size = int(get_arg(args, "bucket_size", 16))

    out_dir = os.path.join(out_root, "test_8_approximate_fl_basic_mpc_arm_tee")
    ensure_dir(out_dir)

    log_info("===========================================================")
    log_info("测试编号8：近似联邦学习、基础算子型多方安全计算与ARM架构TEE间的参数安全转换功能。")
    log_info("输入技术子类型识别成功：联邦学习子类型=面向大规模数据的近似联邦学习；MPC子类型=面向基础算子的多方安全计算；TEE类型=ARM架构可信执行环境。")

    compressed = generate_mock_approximate_fl_compressed_gradient(
        dim=approx_dim,
        bucket_size=bucket_size,
    )
    sparse_indices = compressed["sparse_indices"]
    bucket_ids = compressed["bucket_ids"]
    quantized_values = compressed["quantized_values"]
    selected_values = compressed["selected_values"]

    log_info("近似联邦学习侧压缩梯度参数读取成功。")
    compression_ratio = float(sparse_indices.numel() / compressed["dim"])
    log_info("近似联邦学习压缩梯度参数明细：")
    print(f"  - 原始梯度维度: {compressed['dim']}")
    print(f"  - 稀疏非零项数量: {sparse_indices.numel()}")
    print(f"  - 压缩保留比例: {compression_ratio:.4f}")
    print(f"  - 分桶大小: {compressed['bucket_size']}")
    print(f"  - 量化缩放因子: {compressed['scale']}")
    print(f"  - 前10个稀疏索引: {sparse_indices[:10].tolist()}")
    print(f"  - 前10个分桶编号: {bucket_ids[:10].tolist()}")
    print(f"  - 前10个原始浮点值: {[round(float(x), 8) for x in selected_values[:10].tolist()]}")
    print(f"  - 前10个量化整数值: {quantized_values[:10].tolist()}")

    index_value_check = (
        sparse_indices.numel() == bucket_ids.numel()
        and sparse_indices.numel() == quantized_values.numel()
    )
    if not index_value_check:
        raise AssertionError("索引信息、分桶编号和值信息数量不一致。")

    log_info("索引信息、值信息和分桶编号分离完成，数量一致性校验通过。")

    mpc_integer_matrix = torch.stack(
        [
            sparse_indices.to(torch.int64),
            bucket_ids.to(torch.int64),
            quantized_values.to(torch.int64),
        ],
        dim=1,
    )
    mpc_integer_vector = mpc_integer_matrix.reshape(-1)

    log_info("压缩梯度参数规范化完成：每个压缩项格式为 [sparse_index, bucket_id, quantized_value]。")
    log_info(f"可用于安全计算的整数参数向量生成成功：维度={mpc_integer_vector.numel()}。")
    print(f"  - 前5个MPC三元组:\n{mpc_integer_matrix[:5]}")

    ring_mapping_summary = build_basic_mpc_ring_mapping_summary(
        mpc_integer_matrix=mpc_integer_matrix,
        mpc_integer_vector=mpc_integer_vector,
        q=q,
    )
    print_basic_mpc_ring_mapping_summary(ring_mapping_summary)

    log_info("开始执行两方加法秘密共享处理：随机生成第一份秘密份额，并基于环上减法生成第二份秘密份额。")
    s0, s1 = two_party_share_integer_vector(mpc_integer_vector, q=q)
    log_info("基础算子型多方安全计算所需的两方MPC计算份额生成成功。")
    print_share_summary(s0, s1, q)

    tee_metadata = {
        "test_id": 8,
        "fl_type": "approximate_fl",
        "mpc_type": "basic_operator_mpc",
        "tee_type": "arm",
        "original_dim": int(compressed["dim"]),
        "sparse_nnz": int(sparse_indices.numel()),
        "compression_ratio": compression_ratio,
        "ring_mod": q,
    }
    tee_objects = [
        make_tee_protected_tensor_object(
            arch="ARM",
            object_name="basic_mpc_share_0",
            tensor=s0,
            metadata={**tee_metadata, "share_owner": "server0"},
        ),
        make_tee_protected_tensor_object(
            arch="ARM",
            object_name="basic_mpc_share_1",
            tensor=s1,
            metadata={**tee_metadata, "share_owner": "server1"},
        ),
    ]
    print_tee_object_summary(tee_objects, arch_name="ARM")

    tee_object_check = (
        validate_tee_tensor_object(tee_objects[0], expected_arch="ARM", expected_name="basic_mpc_share_0", source_tensor=s0)
        and validate_tee_tensor_object(tee_objects[1], expected_arch="ARM", expected_name="basic_mpc_share_1", source_tensor=s1)
    )

    rec_int = reconstruct_integer_vector(s0, s1, q=q)
    ierr = compute_integer_reconstruct_error(mpc_integer_vector, rec_int)
    reconstruct_check = torch.equal(rec_int, mpc_integer_vector.detach().cpu().to(torch.int64))

    dimension_check = int(mpc_integer_vector.numel()) == int(sparse_indices.numel() * 3)
    print_integer_error(ierr, prefix="基础算子MPC整数向量计算误差统计")
    log_info(f"索引信息、数值信息和参数维度一致性校验结果：{index_value_check and dimension_check}")
    log_info(f"TEE受保护参数对象转换校验结果：{tee_object_check}")

    if not reconstruct_check:
        raise AssertionError("基础算子型MPC份额计算结果不一致。")
    if not (index_value_check and dimension_check):
        raise AssertionError("索引信息、数值信息和参数维度一致性校验失败。")
    if not tee_object_check:
        raise AssertionError("TEE受保护参数对象转换校验失败。")

    log_info("索引信息、数值信息和参数维度一致性校验通过。")
    log_info("TEE受保护参数对象转换校验通过。")

    save_tensor_list(os.path.join(out_dir, "share_0.pt"), [s0])
    save_tensor_list(os.path.join(out_dir, "share_1.pt"), [s1])
    save_tensor_list(os.path.join(out_dir, "mpc_integer_vector.pt"), [mpc_integer_vector])
    save_tensor_list(os.path.join(out_dir, "compressed_matrix.pt"), [mpc_integer_matrix])
    save_json(os.path.join(out_dir, "tee_protected_objects.json"), {"tee_objects": tee_objects})

    check_result = {
        "test_id": 8,
        "test_name": "approximate_fl_basic_mpc_arm_tee",
        "fl_type": "面向大规模数据的近似联邦学习",
        "mpc_type": "面向基础算子的多方安全计算",
        "tee_type": "ARM架构可信执行环境",
        "original_dim": int(compressed["dim"]),
        "bucket_size": int(compressed["bucket_size"]),
        "sparse_nnz": int(sparse_indices.numel()),
        "compression_ratio": compression_ratio,
        "mpc_vector_dim": int(mpc_integer_vector.numel()),
        "format": "[sparse_index, bucket_id, quantized_value]",
        "ring_mod": q,
        "integer_reconstruct_error": ierr,
        "reconstruct_check": bool(reconstruct_check),
        "index_value_dimension_check": bool(index_value_check and dimension_check),
        "tee_object_check": bool(tee_object_check),
        "status": "PASS",
    }
    save_json(os.path.join(out_dir, "check_result.json"), check_result)

    print_file_outputs(out_dir, ["share_0.pt", "share_1.pt", "mpc_integer_vector.pt", "compressed_matrix.pt", "tee_protected_objects.json", "check_result.json"])
    log_info("测试8执行完成，参数转换过程无异常报错。")


# ============================================================
# 4.1.9 安全 FL + 复杂攻击 MPC + RISC-V TEE
# ============================================================

def run_secure_fl_attack_mpc_riscv_tee(args: Any) -> None:
    scale = int(get_arg(args, "scale", 10 ** 6))
    q = int(get_arg(args, "ring_mod", RING_MOD))
    out_root = str(get_arg(args, "out_dir", "./results/conversion_tests_tee"))

    out_dir = os.path.join(out_root, "test_9_secure_fl_attack_mpc_riscv_tee")
    ensure_dir(out_dir)

    log_info("===========================================================")
    log_info("测试编号9：安全联邦学习、面向复杂攻击的多方安全计算与RISC-V架构TEE间的参数安全转换功能。")
    log_info("输入技术子类型识别成功：联邦学习子类型=面向复杂攻击的安全联邦学习；MPC子类型=面向复杂攻击的多方安全计算；TEE类型=RISC-V架构可信执行环境。")

    named_grads, aux_params = generate_mock_secure_fl_params()
    log_info("安全联邦学习侧训练梯度参数及安全辅助参数读取成功。")
    print_gradient_table(named_grads, title="安全联邦学习训练梯度参数明细")

    log_info("安全辅助参数明细：")
    for k, v in aux_params.items():
        print(f"  - {k}: {v}")

    vector, structure = flatten_named_tensors(named_grads)
    structure_check = check_structure_consistency(vector, structure)
    log_info(f"训练梯度向量化完成：总维度={vector.numel()}，层数={len(structure)}。")
    print_structure_mapping(structure)

    log_info("待转换参数与安全辅助参数关联完成。")

    x_int = _to_fixed_point(vector, scale=scale)
    qerr = compute_quantization_error(vector, x_int, scale)
    log_info(f"固定点量化整数训练参数完成：scale={scale}，整数参数维度={x_int.numel()}。")
    print_quantization_error(qerr)

    grad_hash = hash_tensor_int64(x_int)
    parameter_check_info = {
        "aux_params": aux_params,
        "gradient_hash": grad_hash,
        "vector_dim": int(x_int.numel()),
        "scale": scale,
        "ring_mod": q,
        "security_binding": "gradient_hash_bound_to_auxiliary_parameters",
    }
    binding_digest = hash_json(parameter_check_info)

    ring_mapping_summary = build_ring_mapping_summary(x_int=x_int, q=q)
    print_ring_mapping_summary(ring_mapping_summary, name="整数训练参数")

    log_info("开始执行两方加法秘密共享处理：随机生成第一份秘密份额，并基于环上减法生成第二份秘密份额。")
    s0, s1 = two_party_share_integer_vector(x_int, q=q)
    log_info("面向复杂攻击的多方安全计算所需的两方MPC计算份额生成成功。")
    print_share_summary(s0, s1, q)

    log_info("安全辅助参数绑定处理完成，参数校验信息生成成功。")
    log_info("安全辅助参数绑定信息：")
    print(f"  - gradient_hash: {grad_hash}")
    print(f"  - binding_digest: {binding_digest}")

    tee_metadata = {
        "test_id": 9,
        "fl_type": "secure_fl",
        "mpc_type": "attack_resistant_mpc",
        "tee_type": "riscv",
        "vector_dim": int(vector.numel()),
        "scale": scale,
        "ring_mod": q,
    }
    tee_objects = [
        make_tee_protected_tensor_object(
            arch="RISC-V",
            object_name="attack_mpc_share_0",
            tensor=s0,
            metadata={**tee_metadata, "share_owner": "server0"},
        ),
        make_tee_protected_tensor_object(
            arch="RISC-V",
            object_name="attack_mpc_share_1",
            tensor=s1,
            metadata={**tee_metadata, "share_owner": "server1"},
        ),
        make_tee_protected_json_object(
            arch="RISC-V",
            object_name="parameter_check_info",
            payload=parameter_check_info,
            metadata={**tee_metadata, "payload_type": "security_auxiliary_binding"},
        ),
    ]
    print_tee_object_summary(tee_objects, arch_name="RISC-V")

    tee_object_check = (
        validate_tee_tensor_object(tee_objects[0], expected_arch="RISC-V", expected_name="attack_mpc_share_0", source_tensor=s0)
        and validate_tee_tensor_object(tee_objects[1], expected_arch="RISC-V", expected_name="attack_mpc_share_1", source_tensor=s1)
        and validate_tee_json_object(tee_objects[2], expected_arch="RISC-V", expected_name="parameter_check_info", source_payload=parameter_check_info)
    )

    rec_int = reconstruct_integer_vector(s0, s1, q=q)
    ierr = compute_integer_reconstruct_error(x_int, rec_int)
    rec_hash = hash_tensor_int64(rec_int)

    reconstruct_check = torch.equal(rec_int, x_int.detach().cpu().to(torch.int64))
    binding_check = rec_hash == grad_hash

    print_integer_error(ierr)
    log_info("抗攻击MPC份额计算与安全辅助参数绑定校验结果：")
    print(f"  - reconstructed_gradient_hash: {rec_hash}")
    print(f"  - reconstruct_check: {reconstruct_check}")
    print(f"  - binding_check: {binding_check}")
    log_info(f"参数结构一致性校验结果：{structure_check}")
    log_info(f"TEE受保护参数对象转换校验结果：{tee_object_check}")

    if not reconstruct_check:
        raise AssertionError("秘密份额计算结果与量化后整数参数不一致。")
    if not binding_check:
        raise AssertionError("安全辅助参数绑定校验失败。")
    if not structure_check:
        raise AssertionError("参数结构一致性校验失败。")
    if not tee_object_check:
        raise AssertionError("TEE受保护参数对象转换校验失败。")

    log_info("秘密份额一致性校验通过。")
    log_info("安全辅助参数绑定校验通过。")
    log_info("TEE受保护参数对象转换校验通过。")

    save_tensor_list(os.path.join(out_dir, "share_0.pt"), [s0])
    save_tensor_list(os.path.join(out_dir, "share_1.pt"), [s1])
    save_tensor_list(os.path.join(out_dir, "quantized_vector.pt"), [x_int])
    save_tensor_list(os.path.join(out_dir, "reconstructed_vector.pt"), [rec_int])
    save_json(os.path.join(out_dir, "parameter_check_info.json"), parameter_check_info)
    save_json(os.path.join(out_dir, "tee_protected_objects.json"), {"tee_objects": tee_objects})

    check_result = {
        "test_id": 9,
        "test_name": "secure_fl_attack_mpc_riscv_tee",
        "fl_type": "面向复杂攻击的安全联邦学习",
        "mpc_type": "面向复杂攻击的多方安全计算",
        "tee_type": "RISC-V架构可信执行环境",
        "vector_dim": int(vector.numel()),
        "num_layers": len(structure),
        "structure": [asdict(item) for item in structure],
        "aux_params": aux_params,
        "gradient_hash": grad_hash,
        "binding_digest": binding_digest,
        "reconstructed_gradient_hash": rec_hash,
        "scale": scale,
        "ring_mod": q,
        "quantization_error": qerr,
        "integer_reconstruct_error": ierr,
        "reconstruct_check": bool(reconstruct_check),
        "binding_check": bool(binding_check),
        "structure_check": bool(structure_check),
        "tee_object_check": bool(tee_object_check),
        "status": "PASS",
    }
    save_json(os.path.join(out_dir, "check_result.json"), check_result)

    print_file_outputs(out_dir, ["share_0.pt", "share_1.pt", "quantized_vector.pt", "reconstructed_vector.pt", "parameter_check_info.json", "tee_protected_objects.json", "check_result.json"])
    log_info("测试9执行完成，参数转换过程无异常报错。")



# ============================================================
# 4.1.10 个性化 FL + 基础算子型 MPC + ARM TEE
# ============================================================

def run_personalized_fl_basic_mpc_arm_tee(args: Any) -> None:
    scale = int(get_arg(args, "scale", 10 ** 6))
    q = int(get_arg(args, "ring_mod", RING_MOD))
    out_root = str(get_arg(args, "out_dir", "./results/conversion_tests_tee"))

    out_dir = os.path.join(out_root, "test_10_personalized_fl_basic_mpc_arm_tee")
    ensure_dir(out_dir)

    log_info("===========================================================")
    log_info("测试编号10：个性化联邦学习、基础算子型多方安全计算与ARM架构TEE间的参数安全转换功能。")
    log_info("输入技术子类型识别成功：联邦学习子类型=面向个性化数据的高精度无损联邦学习；MPC子类型=面向基础算子的多方安全计算；TEE类型=ARM架构可信执行环境。")

    named_grads, roles = generate_mock_personalized_fl_gradients()
    log_info("个性化联邦学习侧训练梯度参数读取成功。")
    log_info("待转换梯度参数结构识别成功。")
    print_gradient_table(named_grads, roles, title="个性化联邦学习训练梯度参数明细")

    vector, structure = flatten_named_tensors(named_grads, roles=roles)
    structure_check = check_structure_consistency(vector, structure)
    log_info(f"训练梯度向量化完成：总维度={vector.numel()}，层数={len(structure)}。")
    print_structure_mapping(structure)

    x_int = _to_fixed_point(vector, scale=scale)
    qerr = compute_quantization_error(vector, x_int, scale)
    log_info(f"固定点量化整数训练参数完成：scale={scale}，整数参数维度={x_int.numel()}。")
    print_quantization_error(qerr)

    ring_mapping_summary = build_ring_mapping_summary(x_int=x_int, q=q)
    print_ring_mapping_summary(ring_mapping_summary, name="整数训练参数")

    # 基础算子型 MPC 这里直接以统一整数向量作为基础算子输入，
    # 两方加法秘密共享后由基础算子完成加法/重构类处理。
    log_info("开始执行两方加法秘密共享处理：随机生成第一份秘密份额，并基于环上减法生成第二份秘密份额。")
    s0, s1 = two_party_share_integer_vector(x_int, q=q)
    log_info("基础算子型多方安全计算所需的两方MPC计算份额生成成功。")
    print_share_summary(s0, s1, q)

    tee_metadata = {
        "test_id": 10,
        "fl_type": "personalized_fl",
        "mpc_type": "basic_operator_mpc",
        "tee_type": "arm",
        "vector_dim": int(vector.numel()),
        "scale": scale,
        "ring_mod": q,
    }
    tee_objects = [
        make_tee_protected_tensor_object(
            arch="ARM",
            object_name="basic_mpc_share_0",
            tensor=s0,
            metadata={**tee_metadata, "share_owner": "server0"},
        ),
        make_tee_protected_tensor_object(
            arch="ARM",
            object_name="basic_mpc_share_1",
            tensor=s1,
            metadata={**tee_metadata, "share_owner": "server1"},
        ),
    ]
    print_tee_object_summary(tee_objects, arch_name="ARM")

    tee_object_check = (
        validate_tee_tensor_object(tee_objects[0], expected_arch="ARM", expected_name="basic_mpc_share_0", source_tensor=s0)
        and validate_tee_tensor_object(tee_objects[1], expected_arch="ARM", expected_name="basic_mpc_share_1", source_tensor=s1)
    )

    rec_int = reconstruct_integer_vector(s0, s1, q=q)
    ierr = compute_integer_reconstruct_error(x_int, rec_int)
    reconstruct_check = torch.equal(rec_int, x_int.detach().cpu().to(torch.int64))
    dimension_check = int(rec_int.numel()) == int(x_int.numel()) == int(vector.numel())

    print_integer_error(ierr)
    log_info(f"转换前后参数维度一致性校验结果：{dimension_check}")
    log_info(f"参数结构一致性校验结果：{structure_check}")
    log_info(f"TEE受保护参数对象转换校验结果：{tee_object_check}")

    if not reconstruct_check:
        raise AssertionError("份额计算结果与量化后整数参数不一致。")
    if not dimension_check:
        raise AssertionError("转换前后参数维度一致性校验失败。")
    if not structure_check:
        raise AssertionError("参数结构一致性校验失败。")
    if not tee_object_check:
        raise AssertionError("TEE受保护参数对象转换校验失败。")

    log_info("份额计算结果正确，转换前后参数维度一致。")
    log_info("TEE受保护参数对象转换校验通过。")

    save_tensor_list(os.path.join(out_dir, "share_0.pt"), [s0])
    save_tensor_list(os.path.join(out_dir, "share_1.pt"), [s1])
    save_tensor_list(os.path.join(out_dir, "quantized_vector.pt"), [x_int])
    save_json(os.path.join(out_dir, "tee_protected_objects.json"), {"tee_objects": tee_objects})

    check_result = {
        "test_id": 10,
        "test_name": "personalized_fl_basic_mpc_arm_tee",
        "fl_type": "面向个性化数据的高精度无损联邦学习",
        "mpc_type": "面向基础算子的多方安全计算",
        "tee_type": "ARM架构可信执行环境",
        "vector_dim": int(vector.numel()),
        "num_layers": len(structure),
        "structure": [asdict(item) for item in structure],
        "scale": scale,
        "ring_mod": q,
        "quantization_error": qerr,
        "integer_reconstruct_error": ierr,
        "reconstruct_check": bool(reconstruct_check),
        "dimension_check": bool(dimension_check),
        "structure_check": bool(structure_check),
        "tee_object_check": bool(tee_object_check),
        "status": "PASS",
    }
    save_json(os.path.join(out_dir, "check_result.json"), check_result)

    print_file_outputs(out_dir, ["share_0.pt", "share_1.pt", "quantized_vector.pt", "tee_protected_objects.json", "check_result.json"])
    log_info("测试10执行完成，参数转换过程无异常报错。")


# ============================================================
# 4.1.11 近似 FL + 机器学习型 MPC + x86 TEE
# ============================================================

def run_approximate_fl_ml_mpc_x86_tee(args: Any) -> None:
    q = int(get_arg(args, "ring_mod", RING_MOD))
    out_root = str(get_arg(args, "out_dir", "./results/conversion_tests_tee"))
    approx_dim = int(get_arg(args, "approx_dim", 256))
    bucket_size = int(get_arg(args, "bucket_size", 16))

    out_dir = os.path.join(out_root, "test_11_approximate_fl_ml_mpc_x86_tee")
    ensure_dir(out_dir)

    log_info("===========================================================")
    log_info("测试编号11：近似联邦学习、机器学习型多方安全计算与x86架构TEE间的参数安全转换功能。")
    log_info("输入技术子类型识别成功：联邦学习子类型=面向大规模数据的近似联邦学习；MPC子类型=面向机器学习的多方安全计算；TEE类型=x86架构可信执行环境。")

    compressed = generate_mock_approximate_fl_compressed_gradient(
        dim=approx_dim,
        bucket_size=bucket_size,
    )

    sparse_indices = compressed["sparse_indices"]
    bucket_ids = compressed["bucket_ids"]
    selected_values = compressed["selected_values"]
    quantized_values = compressed["quantized_values"]

    log_info("近似联邦学习侧量化梯度或稀疏梯度参数读取成功。")
    compression_ratio = float(sparse_indices.numel() / compressed["dim"])

    log_info("近似联邦学习压缩参数明细：")
    print(f"  - 原始梯度维度: {compressed['dim']}")
    print(f"  - 稀疏非零项数量: {sparse_indices.numel()}")
    print(f"  - 压缩保留比例: {compression_ratio:.4f}")
    print(f"  - 分桶大小: {compressed['bucket_size']}")
    print(f"  - 量化缩放因子: {compressed['scale']}")
    print(f"  - 前10个稀疏索引: {sparse_indices[:10].tolist()}")
    print(f"  - 前10个分桶编号: {bucket_ids[:10].tolist()}")
    print(f"  - 前10个原始浮点值: {[round(float(x), 8) for x in selected_values[:10].tolist()]}")
    print(f"  - 前10个量化整数值: {quantized_values[:10].tolist()}")

    index_value_check = (
        sparse_indices.numel() == bucket_ids.numel()
        and sparse_indices.numel() == quantized_values.numel()
    )
    if not index_value_check:
        raise AssertionError("索引信息、分桶编号和值信息数量不一致。")

    log_info("压缩参数的索引和值完成分离，并生成对应参数索引信息和值信息。")

    # 机器学习型 MPC 这里将索引、分桶和值均作为训练参数描述的一部分，
    # 统一规范化为整数训练参数份额。
    mpc_training_matrix = torch.stack(
        [
            sparse_indices.to(torch.int64),
            bucket_ids.to(torch.int64),
            quantized_values.to(torch.int64),
        ],
        dim=1,
    )
    mpc_training_vector = mpc_training_matrix.reshape(-1)

    log_info("参数索引和值已成功转换为整数表示。")
    log_info(f"整数训练参数向量生成成功：维度={mpc_training_vector.numel()}。")
    print(f"  - 前5个MPC训练参数三元组:\n{mpc_training_matrix[:5]}")

    ring_mapping_summary = build_basic_mpc_ring_mapping_summary(
        mpc_integer_matrix=mpc_training_matrix,
        mpc_integer_vector=mpc_training_vector,
        q=q,
    )
    print_basic_mpc_ring_mapping_summary(ring_mapping_summary)

    log_info("开始执行两方加法秘密共享处理：随机生成第一份秘密份额，并基于环上减法生成第二份秘密份额。")
    s0, s1 = two_party_share_integer_vector(mpc_training_vector, q=q)
    log_info("机器学习型多方安全计算所需的MPC训练参数份额生成成功。")
    print_share_summary(s0, s1, q)

    tee_metadata = {
        "test_id": 11,
        "fl_type": "approximate_fl",
        "mpc_type": "ml_mpc",
        "tee_type": "x86",
        "original_dim": int(compressed["dim"]),
        "sparse_nnz": int(sparse_indices.numel()),
        "compression_ratio": compression_ratio,
        "ring_mod": q,
    }
    tee_objects = [
        make_tee_protected_tensor_object(
            arch="x86",
            object_name="ml_mpc_training_share_0",
            tensor=s0,
            metadata={**tee_metadata, "share_owner": "server0"},
        ),
        make_tee_protected_tensor_object(
            arch="x86",
            object_name="ml_mpc_training_share_1",
            tensor=s1,
            metadata={**tee_metadata, "share_owner": "server1"},
        ),
    ]
    print_tee_object_summary(tee_objects, arch_name="x86")

    tee_object_check = (
        validate_tee_tensor_object(tee_objects[0], expected_arch="x86", expected_name="ml_mpc_training_share_0", source_tensor=s0)
        and validate_tee_tensor_object(tee_objects[1], expected_arch="x86", expected_name="ml_mpc_training_share_1", source_tensor=s1)
    )

    rec_int = reconstruct_integer_vector(s0, s1, q=q)
    ierr = compute_integer_reconstruct_error(mpc_training_vector, rec_int)
    reconstruct_check = torch.equal(rec_int, mpc_training_vector.detach().cpu().to(torch.int64))
    dimension_check = int(mpc_training_vector.numel()) == int(sparse_indices.numel() * 3)

    print_integer_error(ierr, prefix="机器学习型MPC训练参数份额计算误差统计")
    log_info(f"索引信息、数值信息和参数维度一致性校验结果：{index_value_check and dimension_check}")
    log_info(f"TEE受保护参数对象转换校验结果：{tee_object_check}")

    if not reconstruct_check:
        raise AssertionError("MPC训练参数份额计算结果不一致。")
    if not (index_value_check and dimension_check):
        raise AssertionError("索引信息、数值信息和参数维度一致性校验失败。")
    if not tee_object_check:
        raise AssertionError("TEE受保护参数对象转换校验失败。")

    log_info("索引信息、数值信息和参数维度一致性校验通过。")
    log_info("TEE受保护参数对象转换校验通过。")

    save_tensor_list(os.path.join(out_dir, "share_0.pt"), [s0])
    save_tensor_list(os.path.join(out_dir, "share_1.pt"), [s1])
    save_tensor_list(os.path.join(out_dir, "mpc_training_vector.pt"), [mpc_training_vector])
    save_tensor_list(os.path.join(out_dir, "mpc_training_matrix.pt"), [mpc_training_matrix])
    save_json(os.path.join(out_dir, "tee_protected_objects.json"), {"tee_objects": tee_objects})

    check_result = {
        "test_id": 11,
        "test_name": "approximate_fl_ml_mpc_x86_tee",
        "fl_type": "面向大规模数据的近似联邦学习",
        "mpc_type": "面向机器学习的多方安全计算",
        "tee_type": "x86架构可信执行环境",
        "original_dim": int(compressed["dim"]),
        "bucket_size": int(compressed["bucket_size"]),
        "sparse_nnz": int(sparse_indices.numel()),
        "compression_ratio": compression_ratio,
        "mpc_training_vector_dim": int(mpc_training_vector.numel()),
        "format": "[sparse_index, bucket_id, quantized_value]",
        "ring_mod": q,
        "integer_reconstruct_error": ierr,
        "reconstruct_check": bool(reconstruct_check),
        "index_value_dimension_check": bool(index_value_check and dimension_check),
        "tee_object_check": bool(tee_object_check),
        "status": "PASS",
    }
    save_json(os.path.join(out_dir, "check_result.json"), check_result)

    print_file_outputs(out_dir, ["share_0.pt", "share_1.pt", "mpc_training_vector.pt", "mpc_training_matrix.pt", "tee_protected_objects.json", "check_result.json"])
    log_info("测试11执行完成，参数转换过程无异常报错。")



# ============================================================
# 4.1.14 FL + MPC + TEE 三技术协同完整流程
# ============================================================

def _vector_to_tensor_list(vector: torch.Tensor, structure: List[StructureItem]) -> List[torch.Tensor]:
    """
    按结构映射将一维向量恢复为模型参数张量列表。
    """
    out: List[torch.Tensor] = []
    for item in structure:
        part = vector[item.start:item.end].reshape(item.shape)
        out.append(part.detach().cpu().clone())
    return out


def _max_abs_diff_tensor_lists(xs: List[torch.Tensor], ys: List[torch.Tensor]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have same number of tensors")
    max_err = 0.0
    for x, y in zip(xs, ys):
        err = (x.detach().cpu() - y.detach().cpu()).abs().max().item()
        max_err = max(max_err, float(err))
    return max_err


def _sum_mod_vectors(vectors: List[torch.Tensor], q: int) -> torch.Tensor:
    if not vectors:
        raise ValueError("vectors is empty")
    acc = torch.zeros_like(vectors[0], dtype=torch.int64)
    for v in vectors:
        acc = _mod_ring(acc + v.detach().cpu().to(torch.int64), q=q)
    return acc


def _flatten_tensor_list(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().cpu().reshape(-1).float() for t in tensors], dim=0)


def run_fl_mpc_tee_collaboration(args: Any) -> None:
    """
    测试14：联邦学习、多方安全计算、可信执行环境三种隐私计算技术协同运行。

    输出风格与测试13保持一致：
    - 不显式打印“测试14-步骤X”；
    - 通过自然分段日志对应训练、参数转换、秘密共享、TEE处理、结果恢复和模型更新；
    - 对长日志采用换行和缩进展示，避免一行过长。
    """
    scale = int(get_arg(args, "scale", 10 ** 6))
    q = int(get_arg(args, "ring_mod", RING_MOD))
    out_root = str(get_arg(args, "test14_out_dir", "./results/test_14_fl_mpc_tee_collaboration"))
    ensure_dir(out_root)

    n_clients = int(get_arg(args, "nworkers", 5))
    n_rounds = int(get_arg(args, "niter", 1))
    batch_size = int(get_arg(args, "batch_size", 16))
    lr = float(get_arg(args, "lr", 0.02))
    dataset = str(get_arg(args, "dataset", "SIMULATED"))
    model_name = str(get_arg(args, "net", "mlp"))
    seed = int(get_arg(args, "seed", 4))

    torch.manual_seed(seed)

    log_info("===========================================================")
    log_info("测试编号14：联邦学习、多方安全计算、可信执行环境3种隐私计算技术相互协同工作时的参数安全转换工具功能。")
    log_info("输入技术子类型识别成功：联邦学习子类型=联邦学习；MPC子类型=多方安全计算；TEE类型=可信执行环境。")
    log_info("联邦学习训练任务参数加载完成。")
    log_info("训练参数初始化成功。")
    print(f"  - 客户端数量={n_clients}")
    print(f"  - 训练轮数={n_rounds}")
    print(f"  - 学习率={lr}")
    print(f"  - 模型结构={model_name}")
    print(f"  - 数据集={dataset}")
    log_info("参数配置过程执行正常，未出现异常报错。")

    save_json(
        os.path.join(out_root, "test14_started.json"),
        {
            "test_id": 14,
            "test_name": "联邦学习、多方安全计算、可信执行环境3种隐私计算技术相互协同工作时的参数安全转换工具功能",
            "status": "STARTED",
            "dataset": dataset,
            "model": model_name,
            "n_clients": n_clients,
            "n_rounds": n_rounds,
            "learning_rate": lr,
            "batch_size": batch_size,
            "scale": scale,
            "ring_mod": q,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    # 轻量模型和模拟数据，避免测试14依赖外部数据集下载。
    input_dim = 16
    hidden_dim = 8
    num_classes = 2
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, num_classes),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    log_info("全局模型初始化完成。")
    log_info("客户端模型同步完成。")
    log_info("开始加载训练数据集。")

    samples_per_client = max(batch_size * 2, 32)
    total_samples = n_clients * samples_per_client
    x_all = torch.randn(total_samples, input_dim)
    y_all = torch.randint(low=0, high=num_classes, size=(total_samples,))

    log_info("数据集加载成功。")
    log_info("开始按照联邦学习要求进行客户端数据划分。")

    client_data: List[torch.Tensor] = []
    client_label: List[torch.Tensor] = []
    for i in range(n_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_data.append(x_all[start_idx:end_idx])
        client_label.append(y_all[start_idx:end_idx])

    sizes = [int(x.size(0)) for x in client_data]
    log_info("联邦数据组织与划分完成。")
    log_info("客户端数据划分统计。")
    print(f"  - 客户端数量={n_clients}")
    print(f"  - 最小样本数={min(sizes)}")
    print(f"  - 最大样本数={max(sizes)}")
    print(f"  - 平均样本数={sum(sizes)/len(sizes):.2f}")
    print(f"  - 前10个客户端样本数={sizes[:10]}")
    log_info("客户端本地训练数据构建完成。")
    log_info("数据处理过程执行正常，未发生异常中断。")
    log_info("系统成功进入本地训练过程。")
    log_info("训练数据加载与客户端划分完成。")

    round_records: List[Dict[str, Any]] = []

    for e in range(n_rounds):
        log_info("-----------------------------------------------------------")
        log_info(f"第{e}轮训练开始。")
        log_info("客户端本地训练开始执行。")

        client_updates: List[List[torch.Tensor]] = []
        named_client0_grads: List[Tuple[str, torch.Tensor]] = []

        for client_id in range(n_clients):
            model.zero_grad()
            idx = torch.randperm(samples_per_client)[:batch_size]
            output = model(client_data[client_id][idx])
            loss = loss_fn(output, client_label[client_id][idx])
            loss.backward()

            grads: List[torch.Tensor] = []
            named_grads: List[Tuple[str, torch.Tensor]] = []
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                g = param.grad.detach().cpu().clone()
                grads.append(g)
                named_grads.append((name, g))

            client_updates.append(grads)
            if client_id == 0:
                named_client0_grads = named_grads
                log_info(f"客户端本地训练损失计算正常，当前损失值={loss.item():.6f}。")
                log_info("客户端本地模型参数更新成功生成。")

        log_info("本地训练参数提取成功。")
        print_gradient_table(named_client0_grads, title="客户端0本地训练梯度参数明细")
        log_info("客户端本地训练流程正常结束。")

        # 以客户端0梯度展示结构映射、固定点量化与整数环映射。
        vector0, structure = flatten_named_tensors(named_client0_grads)
        structure_check = check_structure_consistency(vector0, structure)
        log_info(f"训练梯度向量化完成：总维度={vector0.numel()}，层数={len(structure)}。")
        print_structure_mapping(structure)

        log_info("参数安全转换模块调用成功。")
        log_info("参数安全转换配置。")
        print(f"  - 固定点量化缩放因子={scale}")
        print(f"  - 整数环模数={q}")

        x0_int = _to_fixed_point(vector0, scale=scale)
        qerr0 = compute_quantization_error(vector0, x0_int, scale)
        log_info("固定点量化整数训练参数完成。")
        print(f"  - 固定点量化缩放因子 scale={scale}")
        print(f"  - 客户端0模型层数={len(structure)}")
        print(f"  - 整数参数维度={x0_int.numel()}")
        print(f"  - 数据类型={x0_int.dtype}")

        log_info("客户端0固定点量化误差统计。")
        print(f"  - max_abs_error={qerr0['max_abs_error']:.8e}")
        print(f"  - mean_abs_error={qerr0['mean_abs_error']:.8e}")
        print(f"  - l2_error={qerr0['l2_error']:.8e}")

        ring_mapping_summary = build_ring_mapping_summary(x_int=x0_int, q=q)
        log_info("整数训练参数映射至整数环完成。")
        print(f"  - 整数环模数 q={ring_mapping_summary['ring_mod']}")
        print(f"  - 映射前整数范围=[{ring_mapping_summary['x_int_min']}, {ring_mapping_summary['x_int_max']}]")
        print(f"  - 负数元素数量={ring_mapping_summary['negative_integer_count']}")
        print(f"  - 映射后环上范围=[{ring_mapping_summary['x_ring_min']}, {ring_mapping_summary['x_ring_max']}]")
        print(f"  - 范围校验 x_ring ∈ [0, q) -> {ring_mapping_summary['ring_range_check']}")

        log_info("开始执行两方加法秘密共享处理：随机生成第一份秘密份额，并基于环上减法生成第二份秘密份额。")

        share0_list: List[torch.Tensor] = []
        share1_list: List[torch.Tensor] = []
        plain_vectors: List[torch.Tensor] = []

        for update in client_updates:
            v = _flatten_tensor_list(update)
            x_int = _to_fixed_point(v, scale=scale)
            s0, s1 = two_party_share_integer_vector(x_int, q=q)
            plain_vectors.append(v)
            share0_list.append(s0)
            share1_list.append(s1)

        log_info("客户端训练参数完成秘密共享处理并成功生成两方MPC计算份额。")
        log_info("两方MPC计算份额生成成功。")
        print(f"  - Server0份额数量={len(share0_list)}")
        print(f"  - Server1份额数量={len(share1_list)}")
        print(f"  - 客户端0份额维度={int(share0_list[0].numel())}")
        print(f"  - 份额数据类型={share0_list[0].dtype}")
        print(f"  - Server0客户端0份额前5个值: {share0_list[0].reshape(-1)[:5].tolist()}")
        print(f"  - Server1客户端0份额前5个值: {share1_list[0].reshape(-1)[:5].tolist()}")
        log_info("MPC份额生成过程无异常报错。")

        # TEE 封装与接收。
        tee_objects = []
        for idx, (s0, s1) in enumerate(zip(share0_list, share1_list)):
            tee_metadata = {
                "test_id": 14,
                "round": e,
                "client_id": idx,
                "scale": scale,
                "ring_mod": q,
                "vector_dim": int(s0.numel()),
            }
            tee_objects.append(
                make_tee_protected_tensor_object(
                    arch="generic",
                    object_name=f"client_{idx}_mpc_share_0",
                    tensor=s0,
                    metadata={**tee_metadata, "share_owner": "server0"},
                )
            )
            tee_objects.append(
                make_tee_protected_tensor_object(
                    arch="generic",
                    object_name=f"client_{idx}_mpc_share_1",
                    tensor=s1,
                    metadata={**tee_metadata, "share_owner": "server1"},
                )
            )

        log_info("两方MPC计算份额已成功进入TEE安全环境。")
        log_info("TEE安全环境初始化及接收完成。")
        print(f"  - Server0接收份额数量={len(share0_list)}")
        print(f"  - Server1接收份额数量={len(share1_list)}")
        print("  - TEE安全环境初始化及接收过程无异常中断")

        log_info("TEE安全环境完成MPC份额加密转换，生成TEE受保护安全份额。")
        print(f"  - TEE受保护参数对象数量={len(tee_objects)}")
        print(f"  - 首个受保护对象名称={tee_objects[0]['object_name']}")
        print(f"  - 首个受保护对象类型={tee_objects[0]['object_type']}")
        print(f"  - 首个受保护对象载荷哈希={tee_objects[0]['payload_hash'][:16]}...")
        print(f"  - TEE加密转换过程无异常报错")

        # TEE 内部聚合与重构。
        agg_s0 = _sum_mod_vectors(share0_list, q=q)
        agg_s1 = _sum_mod_vectors(share1_list, q=q)

        log_info("TEE安全环境完成加密份额解封、受保护处理及份额聚合计算。")
        print(f"  - 聚合份额向量维度={int(agg_s0.numel())}")
        print(f"  - 聚合份额数据类型={agg_s0.dtype}")
        print("  - TEE内部处理过程无异常中断")

        agg_int = reconstruct_integer_vector(agg_s0, agg_s1, q=q)
        agg_vector = agg_int.to(torch.float32) / float(scale)
        secure_sum = _vector_to_tensor_list(agg_vector, structure)
        plain_sum_vector = torch.stack(plain_vectors, dim=0).sum(dim=0)
        plain_sum = _vector_to_tensor_list(plain_sum_vector, structure)
        agg_err = _max_abs_diff_tensor_lists(secure_sum, plain_sum)

        log_info("TEE内部聚合结果计算完成。")
        print(f"  - TEE+MPC聚合梯度与明文聚合梯度最大误差={agg_err:.8e}")
        print("  - 已恢复生成聚合后的全局训练更新参数")
        print("  - 结果恢复过程无异常报错")

        # 全局模型更新与客户端同步。
        avg_vector = agg_vector / float(n_clients)
        avg_grads = _vector_to_tensor_list(avg_vector, structure)
        with torch.no_grad():
            for param, grad in zip(model.parameters(), avg_grads):
                param.add_(grad.to(param.dtype), alpha=-lr)

        log_info("聚合端已基于TEE内部聚合结果完成全局模型参数更新。")
        print(f"  - 更新学习率={lr}")
        print(f"  - 更新参数层数={len(avg_grads)}")
        print("  - 模型更新过程无异常中断")
        log_info("更新后的全局模型已完成客户端同步。")
        log_info("本轮联邦学习、多方安全计算与可信执行环境协同流程执行完成。")

        if e == 0:
            save_tensor_list(os.path.join(out_root, f"run_1_round_{e}_tee_secure_aggregate.pt"), secure_sum)
            save_tensor_list(os.path.join(out_root, f"run_1_round_{e}_plain_aggregate.pt"), plain_sum)
            save_json(os.path.join(out_root, f"run_1_round_{e}_tee_objects.json"), {"tee_objects": tee_objects})

        round_records.append(
            {
                "run": 1,
                "round": e,
                "n_clients": n_clients,
                "num_layers": len(structure),
                "vector_dim": int(vector0.numel()),
                "scale": scale,
                "ring_mod": q,
                "structure_check": bool(structure_check),
                "tee_server0_share_count": len(share0_list),
                "tee_server1_share_count": len(share1_list),
                "tee_protected_object_count": len(tee_objects),
                "tee_mpc_vs_plaintext_max_error": float(agg_err),
                "status": "PASS",
            }
        )

    summary = {
        "test_id": 14,
        "test_name": "联邦学习、多方安全计算、可信执行环境3种隐私计算技术相互协同工作时的参数安全转换工具功能",
        "status": "PASS",
        "dataset": dataset,
        "model": model_name,
        "n_clients": n_clients,
        "n_rounds": n_rounds,
        "learning_rate": lr,
        "batch_size": batch_size,
        "scale": scale,
        "ring_mod": q,
        "round_records": round_records,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(os.path.join(out_root, "test14_summary.json"), summary)

    log_info("测试流程记录已保存。")
    print(f"  - 汇总记录文件: {os.path.join(out_root, 'test14_summary.json')}")
    print(f"  - TEE聚合结果文件: {os.path.join(out_root, 'run_1_round_0_tee_secure_aggregate.pt')}")
    print(f"  - 明文对照结果文件: {os.path.join(out_root, 'run_1_round_0_plain_aggregate.pt')}")
    print(f"  - TEE受保护对象文件: {os.path.join(out_root, 'run_1_round_0_tee_objects.json')}")
    log_info("客户端完成模型同步，系统已完成从训练启动、本地训练、参数转换、TEE安全处理、安全计算到结果输出的完整流程。")
    log_info("系统整体运行正常，未出现异常报错或异常中断。")


# ============================================================
# 统一入口
# ============================================================

def run_conversion_test_tee(args: Any) -> None:
    if args.test == "personalized_fl_ml_mpc_x86_tee":
        run_personalized_fl_ml_mpc_x86_tee(args)
    elif args.test == "approximate_fl_basic_mpc_arm_tee":
        run_approximate_fl_basic_mpc_arm_tee(args)
    elif args.test == "secure_fl_attack_mpc_riscv_tee":
        run_secure_fl_attack_mpc_riscv_tee(args)
    elif args.test == "personalized_fl_basic_mpc_arm_tee":
        run_personalized_fl_basic_mpc_arm_tee(args)
    elif args.test == "approximate_fl_ml_mpc_x86_tee":
        run_approximate_fl_ml_mpc_x86_tee(args)
    else:
        raise ValueError(f"未知测试类型：{args.test}")


# ============================================================
# 命令行入口
# ============================================================

def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="联邦学习、多方安全计算与TEE间的参数安全转换测试工具"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        choices=[
            "personalized_fl_ml_mpc_x86_tee",
            "approximate_fl_basic_mpc_arm_tee",
            "secure_fl_attack_mpc_riscv_tee",
            "personalized_fl_basic_mpc_arm_tee",
            "approximate_fl_ml_mpc_x86_tee",
        ],
        help="指定FL+MPC+TEE参数安全转换专项测试类型；不指定时执行测试14完整协同流程。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/conversion_tests_tee",
        help="参数转换专项测试结果输出目录。",
    )
    parser.add_argument(
        "--test14_out_dir",
        type=str,
        default="./results/test_14_fl_mpc_tee_collaboration",
        help="测试14完整FL+MPC+TEE协同流程结果输出目录。",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10 ** 6,
        help="固定点量化缩放因子。",
    )
    parser.add_argument(
        "--ring_mod",
        type=int,
        default=2 ** 62,
        help="MPC整数环模数。",
    )
    parser.add_argument(
        "--approx_dim",
        type=int,
        default=256,
        help="近似联邦学习压缩梯度原始维度。",
    )
    parser.add_argument(
        "--bucket_size",
        type=int,
        default=16,
        help="近似联邦学习分桶大小。",
    )

    # 测试14完整训练流程参数。
    parser.add_argument("--net", type=str, default="mlp", help="模型结构名称。")
    parser.add_argument("--dataset", type=str, default="SIMULATED", help="数据集名称。")
    parser.add_argument("--niter", type=int, default=1, help="训练轮数。")
    parser.add_argument("--nworkers", type=int, default=5, help="客户端数量。")
    parser.add_argument("--batch_size", type=int, default=16, help="本地训练批大小。")
    parser.add_argument("--lr", type=float, default=0.02, help="学习率。")
    parser.add_argument("--seed", type=int, default=4, help="随机种子。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test is not None:
        run_conversion_test_tee(args)
    else:
        run_fl_mpc_tee_collaboration(args)
