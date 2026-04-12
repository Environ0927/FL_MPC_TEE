from __future__ import print_function

import aggregation_rules
import numpy as np
import random
import argparse
import attacks
import data_loaders

import os
import math
import time
import csv
import subprocess

import torch
import torch.nn as nn
import torch.utils.data
from matplotlib import pyplot as plt
from models.simple_cnn import SimpleCNN1C
from data.mnist_like_loader import load_mnist, load_fmnist, dirichlet_split_indices, make_client_loaders, make_test_loader
from util.metrics import MetricsLogger, eval_metrics, estimate_comm_bytes


from server import (
    secure_aggregate_client_updates,
    plaintext_sum_client_updates, 
    max_abs_diff
)
@torch.no_grad()
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")
    
def setup_from_cfg(cfg):
    data_name = cfg['data']['name'].lower()
    if data_name == 'cifar10':
        train_loader, test_loader = load_cifar10(batch_size=cfg['training']['batch_size'])
    elif data_name == 'mnist':
        train_loader, test_loader = load_mnist(cfg.get('data_dir','./data'))
    elif data_name == 'fmnist':
        train_loader, test_loader = load_fmnist(cfg.get('data_dir','./data'))
    else:
        raise ValueError("data.name must be CIFAR10, MNIST, or FMNIST")

    net = SimpleCNN1C(num_classes=10).to(device)

    idxs = dirichlet_split_indices(
        labels=np.array(train_loader.dataset.targets),
        n_clients=cfg['data'].get('n_clients',20),
        alpha=cfg['data'].get('dir_alpha',0.3),
        seed=cfg.get('seed',42),
    )
    client_loaders = make_client_loaders(train_loader.dataset, idxs, batch_size=cfg['training']['batch_size'])
    return net, client_loaders, test_loader


def parse_args():
    """
    Parses all commandline arguments.
    """
    parser = argparse.ArgumentParser(description="4.1.2.联邦学习与多方安全计算2种隐私计算技术相互协同工作时的参数安全转换工具")

    ### Model and Dataset
    parser.add_argument("--net", help="net", type=str, default="cnn")
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--dataset", help="dataset", type=str, default="MNIST")
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument("--p", help="bias probability of class 1 in server dataset", type=float, default=0.1)

    ### Training
    parser.add_argument("--niter", help="# iterations", type=int, default=100)
    parser.add_argument("--nworkers", help="# workers", type=int, default=30)
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.02)
    parser.add_argument("--gpu", help="no gpu = -1, gpu training otherwise", type=int, default=1)
    parser.add_argument("--seed", help="seed", type=int, default=4)
    parser.add_argument("--nruns", help="number of runs for averaging accuracy", type=int, default=1)
    parser.add_argument("--test_every", help="testing interval", type=int, default=1)

    ### Aggregations
    parser.add_argument("--aggregation", help="aggregation", type=str, default="fedavg")
    
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=6)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="no", choices=["no", "trim_attack", "krum_attack",
                            "scaling_attack", "fltrust_attack", "label_flipping_attack", "min_max_attack", "min_sum_attack"])

    return parser.parse_args()


def get_device(device):
    """
    Selects the device to run the training process on.
    device: -1 to only use cpu, otherwise cuda if available
    """
    if device == -1:
        ctx = torch.device('cpu')
    else:
        ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(ctx)
    return ctx


def get_net(net_type, num_inputs, num_outputs=10):
    """
    Selects the model architecture.
    """
    if net_type == "lr":
        import models.lr as lr
        net = lr.LinearRegression(input_dim=num_inputs, output_dim=num_outputs)
        print("Using Linear Regression model")
    elif net_type == "cnn":
        from models.simple_cnn import SimpleCNN1C
        net = SimpleCNN1C(num_classes=num_outputs)
        print("Using SimpleCNN1C model")
    else:
        raise NotImplementedError(f"Unknown net type: {net_type}")
    return net



def get_byz(byz_type):
    """
    Gets the attack type.
    byz_type: name of the attack
    """
    return attacks.no_byz


def evaluate_accuracy(data_iterator, net, device, trigger, dataset):
    """
    Evaluate the accuracy and backdoor success rate of the model. Fails if model output is NaN.
    data_iterator: test data iterator
    net: model
    device: device used in training and inference
    trigger: boolean if backdoor success rate should be evaluated
    dataset: name of the dataset used in the backdoor attack
    """
    correct = 0
    total = 0
    successful = 0

    net.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_iterator):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net(inputs)

            if not torch.isnan(outputs).any():
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += inputs.shape[0]
            else:
                print("NaN in output of net")
                raise ArithmeticError

            if trigger:     # backdoor attack
                backdoored_inputs, backdoored_targets = attacks.add_backdoor(inputs, targets, dataset)
                backdoored_outputs = net(backdoored_inputs)
                if not torch.isnan(backdoored_outputs).any():
                    _, backdoored_predicted = backdoored_outputs.max(1)
                    successful += backdoored_predicted.eq(backdoored_targets).sum().item()
                else:
                    print("NaN in output of net")
                    raise ArithmeticError

    success_rate = successful / total
    acc = correct / total
    if trigger:
        return acc, success_rate
    else:
        return acc, None


def plot_results(runs_test_accuracy, runs_backdoor_success, test_iterations, niter):
    """
    Plots the evaluation results.
    runs_test_accuracy: accuracy of the model in each iteration specified in test_iterations of every run
    runs_backdoor_success: backdoor success of the model in each iteration specified in test_iterations of every run
    test_iterations: list of iterations the model was evaluated in
    niter: number of iteration the model was trained for
    """
    test_acc_std = []
    test_acc_list = []
    backdoor_success_std = []
    backdoor_success_list = []

    # insert (0,0) as starting point for plot and calculate mean and standard deviation if multiple runs were performed
    if args.nruns == 1:
        runs_test_accuracy = np.insert(runs_test_accuracy, 0, 0, axis=0)
        test_acc_list = runs_test_accuracy
        test_acc_std = [0 for i in range(0, len(runs_test_accuracy))]
    else:
        runs_test_accuracy = np.insert(runs_test_accuracy, 0, 0, axis=1)
        test_acc_std = np.std(runs_test_accuracy, axis=0)
        test_acc_list = np.mean(runs_test_accuracy, axis=0)

    test_iterations.insert(0, 0)
    # Print accuracy and backdoor success rate in array form to console
    print("Test accuracy of runs:")
    print(repr(runs_test_accuracy))
    
    # Determine in which iteration in what run the highest accuracy was achieved.
    # Also print overall mean accuracy and backdoor success rate
    max_index = np.unravel_index(runs_test_accuracy.argmax(), runs_test_accuracy.shape)
    if args.nruns == 1:
        print("Run 1 in iteration %02d had the highest accuracy of %0.4f" % (max_index[0] * 50, runs_test_accuracy.max()))
    else:
        print("Run %02d in iteration %02d had the highest accuracy of %0.4f" % (max_index[0] + 1, max_index[1] * 50, runs_test_accuracy.max()))
        print("The average final accuracy was: %0.4f with an overall average:" % (test_acc_list[-1]))
        print(repr(test_acc_list))
      
    # Generate plot with two axis displaying accuracy and backdoor success rate over the iterations
    
    
    plt.plot(test_iterations, test_acc_list, color='C0')
    plt.fill_between(test_iterations, test_acc_list - test_acc_std, test_acc_list + test_acc_std, color='C0')
    plt.title("Test Accuracy: " + args.net + ", " + args.dataset + ", " + args.aggregation + ", " + args.byz_type + ", nruns " + str(args.nruns))
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.xlim(0, niter)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()


def weight_init(m):
    """
    Initializes the weights of the layer with random values.
    m: the layer which gets initialized
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=2.24)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def main(args):
    """
    The main function that runs the entire training process of the model.
    args: arguments defining hyperparameters
    """

    # setup
    device = get_device(args.gpu)
    num_inputs, num_outputs, num_labels = data_loaders.get_shapes(args.dataset)
    byz = get_byz(args.byz_type)

    # Print all arguments
    paraString = ('dataset: p' + str(args.p) + '_' + str(args.dataset) + ", server_pc: " + str(args.server_pc) + ", bias: " + str(args.bias)
                  + ", nworkers: " + str(args.nworkers) + ", net: " + str(args.net) + ", niter: " + str(args.niter) + ", lr: " + str(args.lr)
                  + ", batch_size: " + str(args.batch_size)
                  + ", aggregation: " + str(args.aggregation)
                  + ", Seed: " + str(args.seed) + ", Test Every: " + str(args.test_every))
    print(paraString)
    log_info("联邦学习训练任务参数加载完成。")
    log_info(
        f"训练参数初始化成功：客户端数量={args.nworkers}，训练轮数={args.niter}，学习率={args.lr}，模型结构={args.net}，数据集={args.dataset}。"
    )
    log_info("参数配置过程执行正常，未出现异常报错。")
    # saving iterations for averaging
    runs_test_accuracy = []
    runs_backdoor_success = []
    test_iterations = []
    backdoor_success_list = []

    # model
    net = get_net(args.net, num_outputs=num_outputs, num_inputs=num_inputs)
    net = net.to(device)
    num_params = torch.cat([xx.reshape((-1, 1)) for xx in net.parameters()], dim=0).size()[0]  # used for FLOD to determine threshold
    # loss
    softmax_cross_entropy = nn.CrossEntropyLoss()

    # perform parameter checks
    if args.server_pc == 0 and (args.aggregation in ["fltrust", "flod", "flare"] or args.byz_type == "fltrust_attack"):
        raise ValueError("Server dataset size cannot be 0 when aggregation is FLTrust, MPC FLTrust, FLOD or attack is fltrust attack")

    if args.dataset == "HAR" and args.nworkers != 30:
        raise ValueError("HAR only works for 30 workers!")

    # 文件名：dataset_model_aggregation_attack_YYYYMMDD-HHMMSS
    exp_name = f"{args.dataset}_{args.net}_{args.aggregation}_{args.byz_type}_{time.strftime('%Y%m%d-%H%M%S')}"
    # 把 paraString 写进 csv 第一行（见 metrics.py 的改动）
    logger = MetricsLogger(out_dir="./results", exp_name=exp_name, para_string=paraString)
    uplink_enc, downlink_enc = "fp32", "fp32"

    # perform multiple runs
    for run in range(1, args.nruns+1):
        grad_list = []
        test_acc_list = []
        test_iterations = []
        backdoor_success_list = []
        server_process = None

        # fix the seeds for deterministic results
        if args.seed > 0:
            args.seed = args.seed + run - 1
            torch.cuda.manual_seed_all(args.seed)
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)

        net.apply(weight_init)  # initialization of model
        log_info("全局模型初始化完成。")
        log_info("客户端模型同步完成。")
        log_info("开始加载训练数据集。")
        train_data, test_data = data_loaders.load_data(args.dataset, args.seed)  # load the data
        log_info("数据集加载成功。")
        log_info("开始按照联邦学习要求进行客户端数据划分。")
        # assign data to the server and clients
        server_data, server_label, each_worker_data, each_worker_label = data_loaders.assign_data(train_data, args.bias, device,
            num_labels=num_labels, num_workers=args.nworkers, server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=args.seed)
        
        print("Data done")
        log_info("联邦数据组织与划分完成。")
        log_info("客户端本地训练数据构建完成。")
        log_info("数据处理过程执行正常，未发生异常中断。")
        log_info("系统成功进入本地训练过程。")
        with torch.no_grad():
            # training
            # s_trust = None
            
            for e in range(args.niter):
                log_info("-----------------------------------------------------------")
                log_info(f"第{e}轮训练开始。")
                log_info("客户端本地训练开始执行。")
                t_round_start = time.time()  # (ADD) 本轮计时开始
                net.train()
                if e == 0: prev = torch.cat([p.data.flatten() for p in net.parameters()]).clone()
                cur  = torch.cat([p.data.flatten() for p in net.parameters()])
                # print(f"[it={e}] ΔW L2 =", (cur - prev).norm().item())
                prev = cur.clone()
                client_time_sum = 0.0
                # perform local training for each worker
                for i in range(args.nworkers):
                    t_cli0 = time.time()  # <--- START per-client timer

                    minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=args.batch_size, replace=False)
                    net.zero_grad()
                    with torch.enable_grad():
                        output = net(each_worker_data[i][minibatch])
                        loss = softmax_cross_entropy(output, each_worker_label[i][minibatch])
                        loss.backward()

                    grad_list.append([param.grad.clone().detach() for param in net.parameters()])
                    if i == 0:
                        log_info(f"客户端本地训练损失计算正常，当前损失值={loss.item():.6f}。")
                        log_info("客户端本地模型参数更新成功生成。")
                    client_time_sum += (time.time() - t_cli0)  # <--- END per-client
                avg_client_comp_s = client_time_sum / max(1, args.nworkers)
                
                
                server_comp_s = 0.0
                # compute server update and append it to the end of the list
                if args.aggregation in ["rainy_tssc"]:
                    t_srv0 = time.time()
                    net.zero_grad()
                    with torch.enable_grad():
                        output = net(server_data)
                        loss = softmax_cross_entropy(output, server_label)
                        loss.backward()
                    server_grads = [torch.clone(p.grad) for p in net.parameters()]
                    server_comp_s += (time.time() - t_srv0)

                    flat = torch.cat([g.reshape(-1) for g in server_grads]).to(device)
                    s_trust = torch.sign(flat); s_trust[s_trust == 0] = 1
                    s_trust = s_trust.to(torch.int8)
                    
                t_srv1 = time.time()    
                # perform the aggregation
                if args.aggregation == "fedavg":
                    data_sizes = [x.size(dim=0) for x in each_worker_data]
                    grad_in = grad_list
                    log_info("本地训练参数提取成功。")
                    log_info("开始将模型参数转换为MPC计算份额。")
                    log_info("参数秘密共享处理开始执行。")
                    # ===== 安全聚合 =====
                    secure_sum, logical_server0, logical_server1 = secure_aggregate_client_updates(
                        grad_in,
                        scale=10**6,
                    )
                    plain_sum = plaintext_sum_client_updates(grad_in)
                    agg_err = max_abs_diff(secure_sum, plain_sum)
                    
                    print("[INFO] 客户端训练梯度已成功转换为两方MPC计算份额。")
                    print(f"[Round {e}] Secure-vs-plaintext max error: {agg_err:.8f}")
                    secure_avg = [g / len(grad_in) for g in secure_sum]
                    with torch.no_grad():
                        for param, agg_grad in zip(net.parameters(), secure_avg):
                            agg_grad = agg_grad.to(param.device, dtype=param.dtype)
                            param.add_(agg_grad, alpha=-args.lr)
                    print(f"[Round {e}] Secure aggregation finished.")
                    print(f"[Round {e}] Server0 stored {len(logical_server0.client_shares)} client shares.")
                    print(f"[Round {e}] Server1 stored {len(logical_server1.client_shares)} client shares.")
                elif args.aggregation == "rainy_tssc":   
                    from aggregation_rules_tssc import rainy_tssc_step
                    # print("s_trust_pos_ratio =", float((s_trust).float().mean().item()))

                    s_trust, stats = rainy_tssc_step(
                        grad_list=grad_list, net=net, device=device, s_trust=s_trust, round_id=e,
                        k_bits=args.k_bits, dp_clip=args.dp_clip, dp_sigma=args.dp_sigma,
                        use_scale=args.use_scale, use_EF=args.use_EF,
                        lambda_mad=args.lambda_mad, tau_override=args.tau_override,
                        w_max=args.w_max, global_lr=args.lr
                    )
                    # stats["s_trust_pos_ratio"] = float((s_trust > 0).float().mean().item())
                    try:
                        logger.log(extra=stats)
                    except Exception:
                        pass
                elif args.aggregation == "rainy":
                    stats = aggregation_rules.rain(
                        grad_list=grad_list, net=net, device=device, lr=args.lr,
                        dp_clip=args.dp_clip, dp_sigma=args.dp_sigma,
                        use_scale=args.use_scale, use_EF=args.use_EF,
                        lambda_mad=args.lambda_mad, tau_override=args.tau_override, w_max=args.w_max,
                    )
                else:
                    raise NotImplementedError
                server_comp_s += (time.time() - t_srv1)
                
                del grad_list
                grad_list = []
                # evaluate the model accuracy
                if (e + 1) % args.test_every == 0:
                    # 你原有的评测：ACC 和（若 scaling_attack）ASR
                    test_accuracy, test_success_rate = evaluate_accuracy(
                        test_data, net, device,
                        args.byz_type == "scaling_attack", args.dataset
                    )
                    test_acc_list.append(test_accuracy)
                    test_iterations.append(e)

                    # 额外：细粒度指标（loss / balanced_acc）
                    m = eval_metrics(net, test_data, device, num_classes=num_outputs)  # loss / acc / bacc
                    # 通信估计（/round）
                    comm_up, comm_down = estimate_comm_bytes(
                        num_params=num_params, n_clients=args.nworkers,
                        uplink_encoding=uplink_enc, downlink_encoding=downlink_enc
                    )
                    per_client_up_kb   = (comm_up / max(1, args.nworkers)) / 1024.0    # 每客户端上行
                    per_client_down_kb = (comm_down) / 1024.0    
                    # 回合耗时
                    elapsed = time.time() - t_round_start

                    # 写 CSV（ASR 无则留空）
                    logger.log(
                        r=e,
                        time_s=elapsed,
                        n_clients=args.nworkers,
                        agg=args.aggregation,
                        test_loss=m["loss"],
                        acc=m["acc"],
                        bacc=m["bacc"],
                        comm_up=comm_up,
                        comm_down=comm_down,
                        asr=(test_success_rate if args.byz_type == "scaling_attack" else None),
                        # ===== 新增 4 个核心指标 =====
                        client_comp_s=avg_client_comp_s,        # 🧩
                        server_comp_s=server_comp_s,            # 🧮
                        client_up_kb=per_client_up_kb,          # 📡
                        client_down_kb=per_client_down_kb,      # 📡
                        server_overall_s=elapsed                # ⏱️
                    )


                    # 你原有的控制台打印（保留）
                    if args.byz_type == "scaling_attack":
                        backdoor_success_list.append(test_success_rate)
                        print("Iteration %02d. Test_acc %0.4f. Backdoor success rate: %0.4f"
                              % (e, test_accuracy, test_success_rate))
                    else:
                        print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))
                        log_info("本轮联邦学习安全训练流程执行完成。")


        # Append accuracy and backdoor success rate to overall runs list
        if len(runs_test_accuracy) > 0:
            runs_test_accuracy = np.vstack([runs_test_accuracy, test_acc_list])
            if args.byz_type == "scaling_attack":
                runs_backdoor_success = np.vstack([runs_backdoor_success, backdoor_success_list])
        else:
            runs_test_accuracy = test_acc_list
            if args.byz_type == "scaling_attack":
                runs_backdoor_success = backdoor_success_list
        if args.byz_type == "scaling_attack":
            print("Run %02d/%02d done with final accuracy: %0.4f and backdoor success rate: %0.4f" % (run, args.nruns, test_acc_list[-1], backdoor_success_list[-1]))
        else:
            print("Run %02d/%02d done with final accuracy: %0.4f" % (run, args.nruns, test_acc_list[-1]))

    # === close csv (ADD) ===
    logger.close()
    print("Saved CSV:", logger.csv_path)
    log_info("系统已完成从训练启动、参数转换、安全计算到结果输出的完整流程。")
    log_info("系统整体运行正常，未出现异常报错或异常中断。")
    del test_acc_list
    test_acc_list = []


if __name__ == "__main__":
    args = parse_args()     # parse arguments
    main(args)      # call main with parsed arguments
