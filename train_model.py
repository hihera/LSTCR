import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from data_loader import CustomerShopDataset, custom_collate_fn
from transformer_model import TransformerModel
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from metrics import accuracy_at_k, precision_at_k, recall_at_k, f1_score_at_k, map_at_k, ndcg_at_k

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_metrics(probs, labels, topk_list=[1, 3, 5, 8, 10, 15, 20]):
    """
    计算并返回多个评估指标
    probs: (N, num_classes)，每条样本在各类别上的概率分布
    labels: (N,) 每条样本的真实类别(整数标签)
    """
    results = {}  # 新增：定义一个字典来保存各项指标

    for k in topk_list:
        acc = accuracy_at_k(probs, labels, k)
        precision = precision_at_k(probs, labels, k)
        recall = recall_at_k(probs, labels, k)
        f1 = f1_score_at_k(precision, recall)
        map_k = map_at_k(probs, labels, k)
        ndcg = ndcg_at_k(probs, labels, k)

        # 将每个指标存进 results 字典
        results[f'Accuracy@{k}'] = acc
        results[f'Precision@{k}'] = precision
        results[f'Recall@{k}'] = recall
        results[f'F1-Score@{k}'] = f1
        results[f'MAP@{k}'] = map_k
        results[f'NDCG@{k}'] = ndcg

    return results  # 把字典返回


def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, log_dir='./logs'):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        #  1) 训练阶段
        model.train()
        total_train_loss = 0.0
        # 打印或记录每个 batch 的 loss
        for batch_idx, batch in enumerate(train_loader):
            customer_embeds, shop_embeds, short_shop_embeds, labels = batch
            # print("=== shop_embeds.shape ===")
            # print(shop_embeds.shape)
            # print("=== custromer_e ===")
            # print(customer_embeds.shape)
            customer_embeds = customer_embeds.to(device)
            shop_embeds = shop_embeds.to(device)
            short_shop_embeds = short_shop_embeds.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 拼接
            customer_embed_expand = customer_embeds.unsqueeze(1).expand(-1, shop_embeds.size(1), -1)
            short_customer_embed_expand = customer_embeds.unsqueeze(1).expand(-1, short_shop_embeds.size(1), -1)
            input_seq = torch.cat((customer_embed_expand, shop_embeds), dim=2)
            short_input_seq = torch.cat((short_customer_embed_expand, short_shop_embeds), dim=2)
            # print("=== input_seq.shape ===")
            # print(input_seq.shape)
            # print("=== input_short_seq.shape ===")
            # print(short_input_seq.shape)

            # 前向
            output = model(input_seq, short_input_seq)
            # output.shape = [B, seq_len, num_classes]
            last_output = output[:, -1, :]  # 只取最后时刻 (B, num_classes)

            loss = criterion(last_output, labels)  # 与训练集保持一致
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"==> Epoch [{epoch + 1}/{epochs}] | Average Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        #  2) 验证阶段
        model.eval()
        total_val_loss = 0.0
        # 请一定在“验证循环”之前先定义这两个列表，以便收集全部 batch 的概率和标签
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                customer_embeds, shop_embeds, short_shop_embeds, labels = batch

                customer_embeds = customer_embeds.to(device)
                shop_embeds = shop_embeds.to(device)
                short_shop_embeds = short_shop_embeds.to(device)
                labels = labels.to(device)

                customer_embed_expand = customer_embeds.unsqueeze(1).expand(-1, shop_embeds.size(1), -1)
                short_customer_embed_expand = customer_embeds.unsqueeze(1).expand(-1, short_shop_embeds.size(1), -1)
                input_seq = torch.cat((customer_embed_expand, shop_embeds), dim=2)
                short_input_seq = torch.cat((short_customer_embed_expand, short_shop_embeds), dim=2)

                # 检查输入形状
                # print(f"input_seq shape: {input_seq.shape}")

                output = model(input_seq, short_input_seq)
                # output.shape = [B, seq_len, num_classes]
                last_output = output[:, -1, :]  # (B, num_classes)

                # 保持和训练一样，用 last_output 算 loss
                loss = criterion(last_output, labels)
                total_val_loss += loss.item()

                # 转为概率分布
                probs = torch.softmax(last_output, dim=-1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"==> Epoch [{epoch + 1}/{epochs}] | Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        # 将所有 batch 的概率与 label 拼接起来
        all_probs = np.concatenate(all_probs, axis=0)  # shape: (N, num_classes)
        all_labels = np.concatenate(all_labels, axis=0)  # shape: (N,)

        # 计算评估指标
        metrics_result = compute_metrics(all_probs, all_labels, topk_list=[1, 3, 5, 8, 10, 15, 20])
        for metric_name, metric_value in metrics_result.items():
            print(f"{metric_name}: {metric_value:.4f}")

    writer.close()

if __name__ == '__main__':
    file_path = "E:/workspace/thesis data/mall/customer_shop_embed_path_new.txt"
    # file_path = "/Users/zhuxiaoxu/Documents/code/heran/customer_shop_embed_path_new.txt"
    # file_path = "/Users/zhuxiaoxu/Documents/code/heran/customer_shop_embed_path_new_h1.txt"
    dataset = CustomerShopDataset(file_path)

    # 划分训练集和验证集
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    print(f"Loaded {len(dataset)} customers' embeddings")

    input_dim = 160
    model_dim = 256
    num_heads = 8
    num_layers = 3
    output_dim = 175
    dropout = 0.1

    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim, dropout)

    train_model(model, train_loader, val_loader, epochs=176, lr=0.001)
    torch.save(model.state_dict(), 'lstcr_model.pth')
