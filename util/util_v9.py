# util/util_v9.py
import numpy as np
import torch
import torch.nn.functional as F
import copy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Subset

def get_output(loader, net, args, latent=False, softmax_output=False):
    net.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(args.device)
            
            features = net(images, latent_output=True)
            final_outputs = net(images, latent_output=False)
            
            if latent:
                outputs = features
            elif softmax_output:
                outputs = F.softmax(final_outputs, dim=1)
            else:
                outputs = final_outputs
            
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    output_whole = np.concatenate(all_outputs) if len(all_outputs) > 0 else np.array([])
    labels_whole = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
    
    return output_whole, labels_whole

def calculate_sub_prototypes(features, labels, args):
    sub_prototypes = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        class_mask = (labels == label)
        class_features = features[class_mask]
        class_features = class_features / (np.linalg.norm(class_features, axis=1, keepdims=True) + 1e-8)
        n_samples = class_features.shape[0]
        if n_samples == 0: continue
        current_k = min(n_samples, args.k_clusters)
        
        if current_k > 0:
            kmeans = KMeans(n_clusters=current_k, random_state=args.seed, n_init=10).fit(class_features)
            centers = kmeans.cluster_centers_
            centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
            sub_prototypes[label] = centers
            
    return sub_prototypes

def noise_detection_and_correction(features, labels, sub_prototypes, model_preds, args):
    num_samples = len(labels)
    if num_samples == 0: return labels, 0, 0.0

    min_similarities = np.zeros(num_samples)
    norm_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    for i in range(num_samples):
        label = labels[i]
        if label in sub_prototypes:
            sims = np.dot(sub_prototypes[label], norm_features[i])
            min_similarities[i] = np.max(sims)
        else:
            min_similarities[i] = 0

    if len(min_similarities) < 2: return labels, 0, 0.0

    gmm = GaussianMixture(n_components=2, random_state=args.seed).fit(min_similarities.reshape(-1, 1))
    gmm_labels = gmm.predict(min_similarities.reshape(-1, 1))
    clean_gmm_label = np.argmax(gmm.means_[:, 0])
    
    clean_indices = np.where(gmm_labels == clean_gmm_label)[0]
    noisy_indices = np.where(gmm_labels != clean_gmm_label)[0]

    client_quality_score = np.mean(min_similarities[clean_indices]) if len(clean_indices) > 0 else 0.0

    dynamic_thresholds = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        class_clean_indices = clean_indices[labels[clean_indices] == label]
        difficulty = np.mean(min_similarities[class_clean_indices] > args.base_threshold) if len(class_clean_indices) > 0 else 0.0
        dynamic_thresholds[label] = args.base_threshold + args.dynamic_threshold_factor * difficulty

    corrected_labels = np.copy(labels)
    corrected_count = 0
    
    model_probs = F.softmax(torch.tensor(model_preds), dim=1).numpy()
    pseudo_labels = np.argmax(model_probs, axis=1)
    pseudo_labels_confidence = np.max(model_probs, axis=1)

    for i in noisy_indices:
        original_label = corrected_labels[i]
        pseudo_label = pseudo_labels[i]
        confidence = pseudo_labels_confidence[i]
        threshold = dynamic_thresholds.get(pseudo_label, 1.0) 

        if confidence > threshold and original_label != pseudo_label:
            corrected_labels[i] = pseudo_label
            corrected_count += 1
            
    if corrected_count > 0:
        print(f"  [Client] Corrected {corrected_count} labels in this epoch.")
        
    return corrected_labels, corrected_count, client_quality_score

def prototype_knowledge_distillation_loss(features, logits, sub_prototypes, args):
    """
    v9: 计算原型知识蒸馏损失
    """
    if not sub_prototypes:
        return torch.tensor(0.0).to(args.device)

    # 1. 准备原型
    proto_labels = sorted(sub_prototypes.keys())
    all_prototypes = [torch.tensor(sub_prototypes[label], dtype=torch.float32).to(args.device) for label in proto_labels]
    
    # 2. 计算教师分布 Q_proto
    norm_features = F.normalize(features, p=2, dim=1)
    
    teacher_logits = torch.zeros(features.shape[0], args.num_classes).to(args.device)
    for i, label in enumerate(proto_labels):
        # 计算与该类所有子原型的相似度，并取最大值
        sims = torch.matmul(norm_features, all_prototypes[i].T)
        max_sims, _ = torch.max(sims, dim=1)
        teacher_logits[:, label] = max_sims
    
    teacher_probs = F.softmax(teacher_logits / args.temp_pkd, dim=1)
    
    # 3. 计算学生分布 P_model
    student_probs = F.softmax(logits / args.temp_pkd, dim=1)
    
    # 4. 计算 KL 散度损失
    loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean') * (args.temp_pkd ** 2)
    
    return loss