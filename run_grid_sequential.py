import os
import yaml
import numpy as np
import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# === 配置路径 ===
template_config_path = "config/icml24/all/ER_EMA,cifar100,m1000mbs64sbs10.yaml"
output_config_dir = "config/grid_sweep_1000/"
log_dir = "logs/grid_sweep_1000/"
script_to_run = "main.py"
results_csv = "results_1000_resnet_thin.csv"

# === 网格参数 ===
# gamma_list = np.round(np.linspace(0.001, 0.0001, 10), 4)
# lam_list = np.round(np.linspace(0.01, 0.0001, 10), 4)

gamma_list = np.round(np.linspace(0.00025, 0.00035, 5), 6)
lam_list   = np.round(np.linspace(0.0030, 0.0040, 5), 6)


# === 创建目录 ===
os.makedirs(output_config_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# === 如果没有 results.csv，先写表头 ===
write_header = not os.path.exists(results_csv)
if write_header:
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['exp_id', 'p_gamma', 'p_lam', 'final_acc'])

# === 读取模板配置 ===
with open(template_config_path, 'r') as f:
    base_config = yaml.safe_load(f)

# === 构造所有实验任务列表 ===
tasks = []
exp_id = 0
for gamma in gamma_list:
    for lam in lam_list:
        config = base_config.copy()
        config["p_gamma"] = float(gamma)
        config["p_lam"] = float(lam)
        config["p_steps"] = config.get("p_steps", 1)

        config_filename = f"ER_EMA,cifar100,gamma{gamma:.4f}_lam{lam:.4f}.yaml"
        config_path = os.path.join(output_config_dir, config_filename)

        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)

        log_path = os.path.join(log_dir, f"log_gamma{gamma:.4f}_lam{lam:.4f}.log")
        tasks.append((exp_id, gamma, lam, config_path, log_path))
        exp_id += 1

# === 定义任务执行函数 ===
def run_task(task, gpu_id):
    exp_id, gamma, lam, config_path, log_path = task
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[🚀 GPU {gpu_id}] Launching Exp {exp_id}: gamma={gamma}, lam={lam}")
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            ["python", script_to_run, "--config", config_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env
        )
        process.wait()

    # 提取 FINAL_ACC
    final_acc = None
    with open(log_path, 'r') as f:
        for line in f:
            if "FINAL_ACC:" in line:
                try:
                    final_acc = float(line.strip().split("FINAL_ACC:")[1])
                except:
                    final_acc = None
                break

    # 写入结果
    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([exp_id, gamma, lam, final_acc])

    print(f"[✅ GPU {gpu_id}] Finished Exp {exp_id} → Acc={final_acc}")
    return exp_id

# === 双卡并发调度 ===
gpu_pool = [4, 3]
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = []
    for idx, task in enumerate(tasks):
        gpu_id = gpu_pool[idx % 2]  # 轮流分配 GPU 4 和 5
        futures.append(executor.submit(run_task, task, gpu_id))

    for future in as_completed(futures):
        _ = future.result()

print("\n✅ All experiments finished.")
