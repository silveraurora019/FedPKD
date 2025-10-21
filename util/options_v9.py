# util/options_v9.py
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # === FedAvg 核心参数 ===
    parser.add_argument('--rounds', type=int, default=900, help="rounds of training (T)")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs (E)")
    parser.add_argument('--frac', type=float, default=0.1, help="fraction of clients to select (C)")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size (B)")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")

    # === 实验环境参数 ===
    parser.add_argument('--num_users', type=int, default=100, help="number of uses: K")
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, default: 0")
    parser.add_argument('--seed', type=int, default=13, help="random seed")
    
    # --- 数据分布与噪声 ---
    parser.add_argument('--iid', type=bool, default=False, help="i.i.d. (True) or non-i.i.d. (False)")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=10)
  
    # --- 噪声参数 ---s
    parser.add_argument('--level_n_system', type=float, default=0.6, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")

    # === v7 的参数 (v9 继承) ===
    parser.add_argument('--k_clusters', type=int, default=1, help="Number of sub-prototypes per class for K-Means clustering.")
    parser.add_argument('--base_threshold', type=float, default=0.8, help="Base cosine similarity threshold to define learning difficulty.")
    parser.add_argument('--dynamic_threshold_factor', type=float, default=0.1, help="Factor to modulate the dynamic confidence threshold.")

    # === v8 的参数 (v9 继承) ===
    parser.add_argument('--aggregation_alpha', type=float, default=0.7, help="Alpha for balancing data size and quality score in server aggregation (0 to 1).")

    # === v9: 新增的原型知识蒸馏参数 ===
    parser.add_argument('--lambda_pkd', type=float, default=0.5, help="Weight for the Prototype Knowledge Distillation loss.")
    parser.add_argument('--temp_pkd', type=float, default=1.0, help="平衡分类任务和特征空间正则化")

    return parser.parse_args()