from __future__ import annotations

from pathlib import Path
import shutil
import sys


MAIN_PATH = Path("main.py")

OLD_IMPORT = """from server_tee import (
    secure_aggregate_client_updates,
    plaintext_sum_client_updates, 
    max_abs_diff
)
"""

NEW_IMPORT = """# 完整训练流程会用到 server_tee 中的安全聚合函数。
# 但 --test 专项测试 7/8/9 不依赖这些函数，因此这里采用可选导入，
# 避免 server_tee.py 缺少相关函数时，程序还未进入 --test 分支就直接退出。
try:
    from server_tee import (
        secure_aggregate_client_updates,
        plaintext_sum_client_updates,
        max_abs_diff,
    )
except ImportError:
    secure_aggregate_client_updates = None
    plaintext_sum_client_updates = None
    max_abs_diff = None
"""

OLD_CALL = """                    # ===== 安全聚合 =====
                    secure_sum, logical_server0, logical_server1 = secure_aggregate_client_updates(
                        grad_in,
                        scale=10**6,
                    )
"""

NEW_CALL = """                    # ===== 安全聚合 =====
                    if secure_aggregate_client_updates is None:
                        raise ImportError(
                            "当前 server_tee.py 未提供 secure_aggregate_client_updates、"
                            "plaintext_sum_client_updates 或 max_abs_diff。"
                            "专项测试请使用 python main.py --test <测试名>；"
                            "如需运行完整训练流程，请在 server_tee.py 中补充安全聚合函数。"
                        )

                    secure_sum, logical_server0, logical_server1 = secure_aggregate_client_updates(
                        grad_in,
                        scale=10**6,
                    )
"""


def main() -> None:
    if not MAIN_PATH.exists():
        print("[ERROR] 当前目录没有 main.py。请在 FL_MPC_TEE 仓库根目录运行本脚本。")
        sys.exit(1)

    text = MAIN_PATH.read_text(encoding="utf-8")

    backup = MAIN_PATH.with_suffix(".py.bak")
    if not backup.exists():
        shutil.copy2(MAIN_PATH, backup)
        print(f"[INFO] 已备份原始文件：{backup}")

    changed = False

    if OLD_IMPORT in text:
        text = text.replace(OLD_IMPORT, NEW_IMPORT, 1)
        changed = True
        print("[INFO] 已将 server_tee 顶部导入改为可选导入。")
    elif "secure_aggregate_client_updates = None" in text:
        print("[INFO] main.py 已经包含可选导入逻辑。")
    else:
        print("[WARN] 未找到标准 server_tee 导入块，请手动检查 main.py 顶部导入。")

    if OLD_CALL in text:
        text = text.replace(OLD_CALL, NEW_CALL, 1)
        changed = True
        print("[INFO] 已为完整训练流程增加 server_tee 函数缺失提示。")
    elif "当前 server_tee.py 未提供 secure_aggregate_client_updates" in text:
        print("[INFO] main.py 已经包含完整训练流程缺失函数提示。")
    else:
        print("[WARN] 未找到标准 secure_aggregate_client_updates 调用块，未修改完整训练流程。")

    if changed:
        MAIN_PATH.write_text(text, encoding="utf-8")
        print("[INFO] main.py 修复完成。现在可以重新运行：")
    else:
        print("[INFO] main.py 没有发生新的修改。")

    print("  python main.py --test personalized_fl_ml_mpc_x86_tee")
    print("  python main.py --test approximate_fl_basic_mpc_arm_tee")
    print("  python main.py --test secure_fl_attack_mpc_riscv_tee")


if __name__ == "__main__":
    main()
