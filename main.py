import os
import torch
import pandas as pd
import numpy as np
import sys
import logging as lg
import datetime as dt
import random as r
import ssl
import wandb

ssl._create_default_https_context = ssl._create_unverified_context

# 强制离线模式，防止网络连接失败
os.environ["WANDB_MODE"] = "offline"

from src.utils.data import get_loaders
from src.utils import name_match
from src.utils.early_stopping import EarlyStopper
from config.parser import Parser
import warnings

warnings.filterwarnings("ignore")
# === (A) 顶部新增导入 ===
from torch.utils.data import ConcatDataset
from src.utils.metrics_kfr_rhl import KFRRHLMeter, OnlineEvalHook, attach_eval_loader

# === (B) 辅助函数 ===
def build_cum_test_loader_for_task(dataloaders, task_id, batch_size=256, num_workers=4):
    # 将 test0..test{task_id} 累加为一个评测集；若不存在，则回退到 'test' 或 'val'
    test_ds_list = []
    for j in range(task_id + 1):
        key = f"test{j}"
        if key in dataloaders:
            test_ds_list.append(dataloaders[key].dataset)
    if test_ds_list:
        probe_ds = ConcatDataset(test_ds_list)
    else:
        if "test" in dataloaders:
            probe_ds = dataloaders["test"].dataset
        elif "val" in dataloaders:
            probe_ds = dataloaders["val"].dataset
        else:
            raise ValueError("未找到测试/验证集，请提供 'test' 或 'val'")
    return attach_eval_loader(probe_ds, batch_size=batch_size, num_workers=num_workers)


def main():
    runs_accs = []
    runs_fgts = []

    parser = Parser()
    args = parser.parse()

    cf = lg.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch = lg.StreamHandler()

    for run_id in range(args.start_seed, args.start_seed + args.n_runs):
        args = parser.parse()
        args.run_id = run_id

        if args.sweep:
            wandb.init()
            for key in wandb.config.keys():
                setattr(args, key, wandb.config[key])
            parser.check_args()
            for key in wandb.config.keys():
                setattr(args, key, wandb.config[key])

        if args.n_runs > 1: args.seed = run_id
        np.random.seed(args.seed)
        r.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if args.learner is not None:
            learner = name_match.learners[args.learner](args)
            if args.resume: learner.resume(args.model_state, args.buffer_state)
        else:
            raise Warning("Please select the desired learner.")

        logfile = f'{args.tag}.log'
        if not os.path.exists(args.logs_root): os.mkdir(args.logs_root)

        ff = lg.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = lg.getLogger()
        fh = lg.FileHandler(os.path.join(args.logs_root, logfile))
        ch.setFormatter(cf)
        fh.setFormatter(ff)
        logger.addHandler(fh)
        logger.addHandler(ch)
        if args.verbose:
            logger.setLevel(lg.DEBUG)
            logger.warning("Running in VERBOSE MODE.")
        else:
            logger.setLevel(lg.INFO)

        lg.info("=" * 60)
        lg.info("=" * 20 + f"RUN N°{run_id} SEED {args.seed}" + "=" * 20)
        lg.info("=" * 60)
        lg.info("Parameters used for this training")
        lg.info("=" * 20)
        lg.info(args)

        dataloaders = get_loaders(args)

        # wandb 初始化（支持离线）
        if not args.no_wandb and not args.sweep:
            try:
                wandb.init(
                    project=f"{args.learner}",
                    config=args.__dict__,
                    settings=wandb.Settings(init_timeout=120)
                )
            except Exception as e:
                print(f"⚠️ wandb.init 异常，切换为 offline 模式：{e}")
                os.environ["WANDB_MODE"] = "offline"
                wandb.init(
                    project=f"{args.learner}",
                    config=args.__dict__,
                    mode="offline"
                )
            print(f"✅ wandb run mode: {wandb.run.mode}")

        # 训练过程
        if args.training_type == 'inc':
            for task_id in range(args.n_tasks):
                # --- 仅当启用时创建 probe/meter/hook ---
                eval_hook = None
                if getattr(args, 'kfr_enable', False):
                    probe_loader = build_cum_test_loader_for_task(
                        dataloaders, task_id,
                        batch_size=getattr(args, 'kfr_probe_batch_size', 256),
                        num_workers=4
                    )
                    K_list = [int(x) for x in str(getattr(args, 'kfr_k', '50,100')).split(',')]
                    eval_interval = int(getattr(args, 'kfr_eval_interval', 20))
                    save_dir = os.path.join(args.logs_root, f"kfr_rhl/{args.tag}/run{args.run_id}/task{task_id}")
                    meter = KFRRHLMeter(K_list_updates=K_list, eval_interval=eval_interval,
                                        dataset_size=len(probe_loader.dataset),
                                        save_dir=save_dir, run_name=f"task{task_id}")
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    eval_hook = OnlineEvalHook(
                        model=learner.model,
                        val_loader=probe_loader,
                        device=device,
                        meter=meter,
                        predict_fn=lambda x: learner.model.logits(x),
                        preprocess_fn=learner.transform_test,
                        use_autocast=torch.cuda.is_available()
                    )

                for e in range(args.epochs):
                    task_name = f"train{task_id}"
                    if args.train:
                        learner.train(
                            dataloader=dataloaders[task_name],
                            task_name=task_name,
                            task_id=task_id,
                            dataloaders=dataloaders,
                            eval_hook = eval_hook  # <<< 传入钩子,不开启则为none
                        )
                    else:
                        model_state = os.path.join(args.ckpt_root, f"{args.tag}/{args.run_id}/ckpt_train{task_id}.pth")
                        mem_idx = int(len(dataloaders['train']) * args.batch_size / args.n_tasks) * (task_id + 1)
                        buffer_state = os.path.join(args.ckpt_root, f"{args.tag}/{args.run_id}/memory_{mem_idx}.pkl")
                        learner.resume(model_state, buffer_state)
                    learner.before_eval()
                    avg_acc, avg_fgt = learner.evaluate(dataloaders, task_id)

                    # 保存每个任务后的模型
                    model_name = f"task_{task_id + 1}.pth"
                    learner.save(model_name)

                    if not args.no_wandb:
                        wandb.log({
                            "avg_acc": avg_acc,
                            "avg_fgt": avg_fgt,
                            "task_id": task_id
                        })
                        if args.wandb_watch:
                            wandb.watch(learner.model, learner.criterion, log="all", log_freq=1)
                    learner.after_eval()
                # --- 仅当启用时汇总 KFR/RHL 并记录 ---
                if getattr(args, 'kfr_enable', False):
                    summ = meter.summary()
                    print(f"[Task {task_id}] KFR mean: {summ['kfr_mean']}, RHL steps: {summ['rhl_steps']}")
                    if not args.no_wandb:
                        for K in K_list:
                            if not np.isnan(summ['kfr_mean'].get(K, float('nan'))):
                                wandb.log({f"task{task_id}/KFR_{K}": summ['kfr_mean'][K]})
                        if summ["rhl_steps"] is not None:
                            wandb.log({f"task{task_id}/RHL_steps": summ["rhl_steps"]})
            learner.save_results()


        elif args.training_type == 'blurry':
            eval_hook = None
            if getattr(args, 'kfr_enable', False):
                if "test" in dataloaders:
                    probe_ds = dataloaders["test"].dataset
                elif "val" in dataloaders:
                    probe_ds = dataloaders["val"].dataset
                else:
                    raise ValueError("未找到测试/验证集")
                probe_loader = attach_eval_loader(probe_ds,
                                                  batch_size=getattr(args, 'kfr_probe_batch_size', 256),
                                                  num_workers=4)
                K_list = [int(x) for x in str(getattr(args, 'kfr_k', '50,100')).split(',')]
                eval_interval = int(getattr(args, 'kfr_eval_interval', 20))
                save_dir = os.path.join(args.logs_root, f"kfr_rhl/{args.tag}/run{args.run_id}")
                meter = KFRRHLMeter(K_list_updates=K_list, eval_interval=eval_interval,
                                    dataset_size=len(probe_loader.dataset),
                                    save_dir=save_dir, run_name="blurry")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                eval_hook = OnlineEvalHook(learner.model, probe_loader, device, meter,
                                           predict_fn=lambda x: learner.model.logits(x),
                                           preprocess_fn=learner.transform_test,
                                           use_autocast=torch.cuda.is_available())
            learner.train(dataloaders['train'], eval_hook=eval_hook)
            avg_acc = learner.evaluate_offline(dataloaders, epoch=1)
            avg_fgt = 0
            if getattr(args, 'kfr_enable', False):
                summ = meter.summary()
                print(f"[blurry] KFR mean: {summ['kfr_mean']}, RHL: {summ['rhl_steps']}")
                if not args.no_wandb:
                    for K in K_list:
                        if not np.isnan(summ['kfr_mean'].get(K, float('nan'))):
                            wandb.log({f"KFR_{K}": summ["kfr_mean"][K]})
                    if summ["rhl_steps"] is not None:
                        wandb.log({"RHL_steps": summ["rhl_steps"]})
            if not args.no_wandb:
                wandb.log({"avg_acc": avg_acc})
                if args.wandb_watch:
                    wandb.watch(learner.model, learner.criterion, log="all", log_freq=1)
            learner.save_results_offline()

        elif args.training_type == 'uni':
            for e in range(args.epochs):
                learner.train(dataloaders['train'], epoch=e)
                avg_acc = learner.evaluate_offline(dataloaders, epoch=e)
                avg_fgt = 0
                if not args.no_wandb:
                    wandb.log({
                        "Accuracy": avg_acc,
                        "loss": learner.loss
                    })
            learner.save_results_offline()

        runs_accs.append(avg_acc)
        runs_fgts.append(avg_fgt)
        # === 新增：打印一行最终结果供脚本抓取 ===
        print(f"FINAL_ACC: {avg_acc:.6f}")  # 用于 run_grid_sweep.py 抓取

        if not args.no_wandb:
            wandb.finish()

    if args.n_runs > 1:
        df_acc = pd.DataFrame(runs_accs)
        df_fgt = pd.DataFrame(runs_fgts)
        results_dir = os.path.join(args.results_root, args.tag)
        lg.info(f"Results for the aggregated runs are save in : {results_dir}")
        df_acc.to_csv(os.path.join(results_dir, 'runs_accs.csv'), index=False)
        df_fgt.to_csv(os.path.join(results_dir, 'runs_fgts.csv'), index=False)

    sys.exit(0)


if __name__ == '__main__':
    main()
