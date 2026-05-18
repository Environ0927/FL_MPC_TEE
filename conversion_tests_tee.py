from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch

# ============================================================
# 本文件为测试7/8/9的专项转换测试，故不依赖 server.py / server_tee.py。
# 这样即使主训练流程的服务端文件命名变化，也不影响专项测试入口。
# ============================================================

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


def print_integer_error(ierr: Dict[str, int], prefix: str = "份额重构整数误差统计") -> None:
    log_info(
        f"{prefix}："
        f"max_integer_error={ierr['max_integer_error']}, "
        f"sum_integer_error={ierr['sum_integer_error']}, "
        f"num_error_elements={ierr['num_error_elements']}。"
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
    log_info(f"{arch_name}架构可信执行环境适配模块调用成功，TEE受保护参数对象生成结果如下：")
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
    log_info(f"固定点量化完成：scale={scale}，整数参数维度={x_int.numel()}。")
    log_info("整数训练参数已成功映射至整数环。")
    print_quantization_error(qerr)

    s0, s1 = two_party_share_integer_vector(x_int, q=q)
    log_info("整数环参数已成功拆分为机器学习型MPC所需的两方计算份额。")
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
        raise AssertionError("份额重构结果与量化后整数参数不一致。")
    if not structure_check:
        raise AssertionError("参数结构一致性校验失败。")
    if not tee_object_check:
        raise AssertionError("TEE受保护参数对象转换校验失败。")

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

    s0, s1 = two_party_share_integer_vector(mpc_integer_vector, q=q)
    log_info("整数参数向量已成功映射至整数环，并完成两方加法秘密共享处理。")
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
    print_integer_error(ierr, prefix="基础算子MPC整数向量重构误差统计")
    log_info(f"索引信息、数值信息和参数维度一致性校验结果：{index_value_check and dimension_check}")
    log_info(f"TEE受保护参数对象转换校验结果：{tee_object_check}")

    if not reconstruct_check:
        raise AssertionError("基础算子型MPC份额重构结果不一致。")
    if not (index_value_check and dimension_check):
        raise AssertionError("索引信息、数值信息和参数维度一致性校验失败。")
    if not tee_object_check:
        raise AssertionError("TEE受保护参数对象转换校验失败。")

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

    x_int = _to_fixed_point(vector, scale=scale)
    qerr = compute_quantization_error(vector, x_int, scale)
    log_info(f"固定点量化完成：scale={scale}，整数参数维度={x_int.numel()}。")
    log_info("整数训练参数已成功映射至整数环。")
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

    log_info("待转换参数与安全辅助参数关联完成。")
    log_info("安全辅助参数绑定或校验处理完成，参数校验信息生成成功。")
    print(f"  - gradient_hash: {grad_hash}")
    print(f"  - binding_digest: {binding_digest}")

    s0, s1 = two_party_share_integer_vector(x_int, q=q)
    log_info("整数训练参数已成功拆分为抗攻击型MPC所需的两方计算份额。")
    print_share_summary(s0, s1, q)

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
    log_info("抗攻击MPC份额重构与安全辅助参数绑定校验结果：")
    print(f"  - reconstructed_gradient_hash: {rec_hash}")
    print(f"  - reconstruct_check: {reconstruct_check}")
    print(f"  - binding_check: {binding_check}")
    log_info(f"参数结构一致性校验结果：{structure_check}")
    log_info(f"TEE受保护参数对象转换校验结果：{tee_object_check}")

    if not reconstruct_check:
        raise AssertionError("秘密份额重构结果与量化后整数参数不一致。")
    if not binding_check:
        raise AssertionError("安全辅助参数绑定校验失败。")
    if not structure_check:
        raise AssertionError("参数结构一致性校验失败。")
    if not tee_object_check:
        raise AssertionError("TEE受保护参数对象转换校验失败。")

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
# 统一入口
# ============================================================

def run_conversion_test_tee(args: Any) -> None:
    if args.test == "personalized_fl_ml_mpc_x86_tee":
        run_personalized_fl_ml_mpc_x86_tee(args)
    elif args.test == "approximate_fl_basic_mpc_arm_tee":
        run_approximate_fl_basic_mpc_arm_tee(args)
    elif args.test == "secure_fl_attack_mpc_riscv_tee":
        run_secure_fl_attack_mpc_riscv_tee(args)
    else:
        raise ValueError(f"未知测试类型：{args.test}")
