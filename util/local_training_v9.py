# util/local_training_v9.py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np 
import copy
from .util_v9 import get_output, calculate_sub_prototypes, noise_detection_and_correction, prototype_knowledge_distillation_loss

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.local_dataset = DatasetSplit(dataset, idxs)
        self.idxs = idxs
        
    def update_weights(self, net, seed, w_g, epoch, mu=0):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        original_targets = np.array([self.local_dataset.dataset.targets[i] for i in self.local_dataset.idxs])
        images = torch.stack([self.local_dataset[i][0] for i in range(len(self.local_dataset))])
        current_labels = torch.tensor(original_targets, dtype=torch.long)
        
        client_quality_score = 0.0
        sub_prototypes = {}

        for iter_ in range(epoch):
            # --- 步骤A: 标签净化与原型生成 ---
            net.eval()
            temp_dataset = TensorDataset(images, current_labels)
            eval_loader = DataLoader(temp_dataset, batch_size=self.args.local_bs, shuffle=False)
            
            features, _ = get_output(eval_loader, net, self.args, latent=True)
            model_preds, _ = get_output(eval_loader, net, self.args, latent=False, softmax_output=False)
            
            sub_prototypes = calculate_sub_prototypes(features, current_labels.numpy(), self.args)
            
            corrected_labels, _, quality_score = noise_detection_and_correction(
                features, current_labels.numpy(), sub_prototypes, model_preds, self.args
            )
            current_labels = torch.tensor(corrected_labels, dtype=torch.long)

            # 在最后一个本地轮次结束后记录质量分数
            if iter_ == epoch - 1:
                client_quality_score = quality_score
                
            # --- 步骤B: 使用新损失函数进行训练 ---
            net.train()
            train_dataset = TensorDataset(images, current_labels)
            ldr_train = DataLoader(train_dataset, batch_size=self.args.local_bs, shuffle=True)
            batch_loss = []

            for batch_idx, (batch_images, batch_labels) in enumerate(ldr_train):
                batch_images, batch_labels = batch_images.to(self.args.device), batch_labels.to(self.args.device)
                
                net.zero_grad()
                
                # 获取特征和logits
                features = net(batch_images, latent_output=True)
                logits = net(batch_images, latent_output=False)
                
                # 1. 计算修正后的交叉熵损失
                loss_ce = self.loss_func(logits, batch_labels)
                
                # 2. 计算原型知识蒸馏损失
                loss_pkd = prototype_knowledge_distillation_loss(features, logits, sub_prototypes, self.args)
                
                # 3. 组合损失
                loss = loss_ce + self.args.lambda_pkd * loss_pkd

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print(f"  [Client] Local Epoch {iter_+1}/{epoch}, Loss: {epoch_loss[-1]:.4f} (CE: {loss_ce.item():.4f}, PKD: {loss_pkd.item():.4f})")

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), client_quality_score


def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc