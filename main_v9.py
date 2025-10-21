# main_v9.py
import os
import copy
import numpy as np
import random
import time
import torch
import torch.nn as nn

# 导入 v9 组件
from util.options_v9 import args_parser
from util.local_training_v9 import LocalUpdate, globaltest
from util.fedavg import FedAvg, FedAvgWeighted
from util.util_v0 import add_noise
from util.dataset import get_dataset
from model.build_model import build_model

np.set_printoptions(threshold=np.inf)

"""
Version 9: FedAvg with Prototype Knowledge Distillation
- Based on v8, with an innovative loss function.
- Local loss = Corrected Cross-Entropy Loss + Prototype Knowledge Distillation Loss (L_PKD).
- L_PKD uses prototypes as a 'local teacher' to regularize the feature space.
"""

if __name__ == '__main__':

    start_time = time.time()

    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    rootpath = "./record/"

    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    if not os.path.exists(rootpath + 'txtsave/'): os.makedirs(rootpath + 'txtsave/')
    txtpath = rootpath + 'txtsave/V9_ProtoKD_lambda_%.2f_%s_%s_NL_%.1f_LB_%.1f_Rnd_%d_E_%d_Frac_%.2f_LR_%.3f_Seed_%d_K_%d' % (
        args.lambda_pkd, args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.rounds, 
        args.local_ep, args.frac, args.lr, args.seed, args.k_clusters)

    if args.iid: txtpath += "_IID"
    else: txtpath += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    f_acc = open(txtpath + '_acc.txt', 'a')

    f_acc.write("="*50 + "\n")
    f_acc.write("Training Parameters (V9 - Prototype Knowledge Distillation):\n")
    f_acc.write(str(args) + "\n")
    f_acc.write("="*50 + "\n")
    f_acc.flush()
    
    netglob = build_model(args)
    if 'resnet' not in args.model:
        print("Warning: This version works best with models that provide good feature embeddings for similarity calculation, like ResNet.")

    max_accuracy = 0.0

    # ============================ Training Loop ============================
    print("\n" + "="*25 + " Stage: FedAvg with V9 Local Loss " + "="*25, flush=True)
    final_accuracies = []
    
    m = max(int(args.frac * args.num_users), 1)

    for rnd in range(args.rounds):
        print(f"\n--- FedAvg Round: {rnd+1}/{args.rounds} ---", flush=True)
        w_locals, local_losses, quality_scores = [], [], []
        
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f"Selected clients for this round: {idxs_users}", flush=True)
        
        for idx in idxs_users:
            print(f"--> Training client {idx}...")
            # 使用 v9 的 LocalUpdate
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            
            w, loss, quality_score = local.update_weights(
                net=copy.deepcopy(netglob).to(args.device), 
                seed=args.seed,
                w_g=netglob.state_dict(),
                epoch=args.local_ep
            )
            w_locals.append({'id': idx, 'w': copy.deepcopy(w)})
            local_losses.append(copy.deepcopy(loss))
            quality_scores.append(quality_score)
            print(f"  [Client {idx}] Final Quality Score: {quality_score:.4f}")

        # === v8 的质量感知聚合 (v9 继承) ===
        client_data_sizes = np.array([len(dict_users[d['id']]) for d in w_locals])
        weights_by_size = client_data_sizes / np.sum(client_data_sizes)
        
        quality_scores = np.array(quality_scores)
        quality_scores_exp = np.exp(quality_scores - np.max(quality_scores))
        weights_by_quality = quality_scores_exp / np.sum(quality_scores_exp)
        
        alpha = args.aggregation_alpha
        final_weights = (1 - alpha) * weights_by_size + alpha * weights_by_quality
        final_weights = final_weights / np.sum(final_weights)

        print(f"Final combined aggregation weights (alpha={alpha}): {np.round(final_weights, 3)}")

        w_glob = FedAvgWeighted([d['w'] for d in w_locals], final_weights)
        netglob.load_state_dict(copy.deepcopy(w_glob))

        # --- 评估 ---
        acc_s3 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        final_accuracies.append(acc_s3)
        max_accuracy = max(max_accuracy, acc_s3)
        
        print(f"Test Accuracy after round {rnd+1}: {acc_s3:.4f}", flush=True)
        f_acc.write(f"round {rnd}, test acc  {acc_s3:.4f} \n"); f_acc.flush()

    # ============================ Final Result Output ============================
    print("\n" + "="*30 + " Final Results " + "="*30, flush=True)
    if len(final_accuracies) >= 10:
        last_10_accuracies = final_accuracies[-10:]
        mean_acc = np.mean(last_10_accuracies)
        var_acc = np.var(last_10_accuracies)
        print(f"Mean of last 10 rounds test accuracy: {mean_acc:.4f}", flush=True)
        print(f"Variance of last 10 rounds test accuracy: {var_acc:.6f}", flush=True)
        f_acc.write(f"\nMean of last 10 rounds test accuracy: {mean_acc:.4f}\n")
        f_acc.write(f"Variance of last 10 rounds test accuracy: {var_acc:.6f}\n")
    elif len(final_accuracies) > 0:
        mean_acc = np.mean(final_accuracies)
        var_acc = np.var(final_accuracies)
        print(f"Mean of final {len(final_accuracies)} rounds test accuracy: {mean_acc:.4f}", flush=True)
        print(f"Variance of final {len(final_accuracies)} rounds test accuracy: {var_acc:.6f}", flush=True)
        f_acc.write(f"\nMean of final {len(final_accuracies)} rounds test accuracy: {mean_acc:.4f}\n")
        f_acc.write(f"Variance of final {len(final_accuracies)} rounds test accuracy: {var_acc:.6f}\n")
    
    print(f"\nMaximum test accuracy achieved: {max_accuracy:.4f}", flush=True)
    f_acc.write(f"\nMaximum test accuracy achieved: {max_accuracy:.4f}\n")

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\nTotal Training Time: {hours}h {minutes}m {seconds}s", flush=True)
    f_acc.write(f"\nTotal Training Time: {hours}h {minutes}m {seconds}s\n")


    f_acc.close()
    torch.cuda.empty_cache()
    print("\nTraining Finished!", flush=True)