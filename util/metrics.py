# util/metrics.py
import os, csv
import torch
import torch.nn.functional as F

class MetricsLogger:
    def __init__(self, out_dir: str, exp_name: str, para_string: str = None):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, f"{exp_name}.csv")
        self.f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        # 先写一行实验设置（来自 main.py 的 paraString）
        if para_string is not None:
            self.w.writerow(["# EXP SETTINGS", para_string])
        # 表头（原有列 + 新增列）
        self.w.writerow([
            "round","time_s","num_clients","agg",
            "test_loss","acc","balanced_acc",
            "comm_up_bytes","comm_down_bytes","asr",
            # ===== 新增 =====
            "client_comp_s",              # 🧩 Per-Client Comp. Cost (s)
            "server_comp_s",              # 🧮 Server-Side Comp. Cost (s)
            "client_up_kb",               # 📡 Per-Client Comm. Cost (KB, uplink)
            "client_down_kb",             # 📡 Per-Client Comm. Cost (KB, downlink)
            "server_overall_s"            # ⏱️ Server-Side Overall Runtime (s)
        ])
        self.f.flush()

    def log(self, r, time_s, n_clients, agg,
            test_loss, acc, bacc,
            comm_up, comm_down, asr=None,
            # ===== 新增参数 =====
            client_comp_s=None, server_comp_s=None,
            client_up_kb=None, client_down_kb=None,
            server_overall_s=None):
        self.w.writerow([
            r, round(time_s,4), n_clients, agg,
            round(test_loss,6), round(acc,6), round(bacc,6),
            int(comm_up), int(comm_down),
            ("" if asr is None else round(asr,6)),
            # ===== 新增写入 =====
            ("" if client_comp_s   is None else round(client_comp_s, 6)),
            ("" if server_comp_s   is None else round(server_comp_s, 6)),
            ("" if client_up_kb    is None else round(client_up_kb, 6)),
            ("" if client_down_kb  is None else round(client_down_kb, 6)),
            ("" if server_overall_s is None else round(server_overall_s, 6))
        ])
        self.f.flush()

    def close(self):
        self.f.close()


@torch.no_grad()
def eval_metrics(model, loader, device, num_classes=10):
    """返回: avg CE loss, accuracy, balanced accuracy"""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    cls_total = [0]*num_classes
    cls_tp    = [0]*num_classes
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        total += y.numel()
        correct += (pred==y).sum().item()
        for c in range(num_classes):
            m = (y==c)
            n = m.sum().item()
            if n>0:
                cls_total[c]+=n
                cls_tp[c]+= (pred[m]==c).sum().item()
    acc = correct/max(1,total)
    avg_loss = loss_sum/max(1,total)
    recalls = [(cls_tp[c]/cls_total[c]) for c in range(num_classes) if cls_total[c]>0]
    bacc = sum(recalls)/len(recalls) if recalls else 0.0
    return {"loss": avg_loss, "acc": acc, "bacc": bacc}


def estimate_comm_bytes(num_params: int, n_clients: int,
                        uplink_encoding="fp32", downlink_encoding="fp32"):
    """
    理论通信估算（/轮）:
      uplink_total  = n_clients × per_client_upload
      downlink_bcast= 1 × broadcast（每个客户端都要收同样的量）
    编码：'fp32'(4B), 'fp16'(2B), 'sign'(1bit=0.125B)
    """
    def bpp(enc):
        return 4 if enc=="fp32" else (2 if enc=="fp16" else 0.125)
    up_total = n_clients * num_params * bpp(uplink_encoding)
    down_bcast = num_params * bpp(downlink_encoding)
    return up_total, down_bcast
