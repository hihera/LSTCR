import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader import CustomerShopDataset, custom_collate_fn
from transformer_model import TransformerModel
import torch.nn.functional as F
from metrics import accuracy_at_k, precision_at_k, recall_at_k, f1_score_at_k, map_at_k, ndcg_at_k

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_metrics(probs, labels, topk_list=[1, 3, 5, 8, 10, 15, 20]):
    """与训练代码保持完全一致的评估指标计算"""
    results = {}
    for k in topk_list:
        results[f'Accuracy@{k}'] = accuracy_at_k(probs, labels, k)
        results[f'Precision@{k}'] = precision_at_k(probs, labels, k)
        results[f'Recall@{k}'] = recall_at_k(probs, labels, k)
        results[f'F1-Score@{k}'] = f1_score_at_k(
            results[f'Precision@{k}'], results[f'Recall@{k}'])
        results[f'MAP@{k}'] = map_at_k(probs, labels, k)
        results[f'NDCG@{k}'] = ndcg_at_k(probs, labels, k)
    return results

def load_feature_from_file(file_path):
    """增加错误处理的特征加载"""
    try:
        return torch.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return torch.zeros(1, 32)  # 返回空特征保持维度

def process_sequence(data_line):
    """改进的数据解析方法，支持长短时序"""
    parts = data_line.strip().split('—>')
    customer = parts[0]
    shops = parts[1:] if len(parts) > 1 else []
    
    # 客户特征处理
    customer_features = [f.split('|')[-1] for f in customer.split('&')]
    customer_embed = torch.cat([load_feature_from_file(p) for p in customer_features], dim=-1)
    
    # 店铺序列处理
    full_sequence = []
    short_sequence = []
    label = None
    
    for i, shop in enumerate(shops):
        # 最后一个作为标签
        if i == len(shops) - 1:  
            label = int(shop.split("&")[6].split("|")[1])
            continue
            
        # 特征提取
        features = [f.split('|')[-1] for f in shop.split('&')]
        embed = torch.cat([load_feature_from_file(p) for p in features], dim=-1)
        
        # 长短序列分离
        full_sequence.append(embed)
        if i >= len(shops)-5:  # 保留最后5个作为短期序列
            short_sequence.append(embed)
    
    return (
        customer_embed.unsqueeze(0), 
        torch.stack(full_sequence).unsqueeze(0),
        torch.stack(short_sequence).unsqueeze(0),
        torch.tensor(label, dtype=torch.long)
        
def predict(model, customer_embed, full_seq, short_seq):
    """支持长短序列的预测方法"""
    model.eval()
    inputs = {
        'customer': customer_embed.to(device),
        'full_seq': full_seq.to(device),
        'short_seq': short_seq.to(device)
    }
    
    with torch.no_grad():
        output = model(**inputs)
    
    probs = F.softmax(output[:, -1, :], dim=1)
    return probs.cpu()

def evaluate(model, test_loader, topk=(1,5)):
    """完整评估流程"""
    all_probs = []
    all_labels = []
    
    for batch in test_loader:
        cust_emb, full_seq, short_seq, labels = batch
        probs = predict(model, cust_emb, full_seq, short_seq)
        
        all_probs.append(probs.numpy())
        all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    return compute_metrics(all_probs, all_labels)

def main():
    """改进的主函数，支持两种模式"""
    # 模型配置（与训练完全一致）
    model_config = {
        'input_dim': 160,
        'model_dim': 256,
        'num_heads': 8,
        'num_layers': 3,
        'output_dim': 175,
        'dropout': 0.1
    }
    
    # 加载训练好的模型
    model = TransformerModel(**model_config)
    model.load_state_dict(torch.load('lstcr_model.pth', map_location=device))
    model.to(device)
    
    # 模式选择：单样本预测或批量评估
    mode = input("Select mode [single/batch]: ").lower()
    
    if mode == 'single':
        # 单样本预测演示
        test_line = "46|8|...（测试数据）"  # 你的测试数据
        customer, full_seq, short_seq, label = process_sequence(test_line)
        
        probs = predict(model, customer, full_seq, short_seq)
        topk_values, topk_indices = torch.topk(probs, 5)
        
        print(f"\n真实标签: {label.item()}")
        print("Top 5预测结果:")
        for i, (val, idx) in enumerate(zip(topk_values[0], topk_indices[0])):
            print(f"{i+1}. 店铺ID: {idx.item()} 概率: {val.item():.4f}")
            
    elif mode == 'batch':
        # 批量评估模式
        test_set = CustomerShopDataset("test_data.txt")  # 测试集路径
        test_loader = DataLoader(test_set, batch_size=64, collate_fn=custom_collate_fn)
        
        metrics = evaluate(model, test_loader)
        
        print("\n=== 测试集评估结果 ===")
        for k in [1,5,10]:
            print(f"Top-{k} Accuracy: {metrics[f'Accuracy@{k}']:.4f}")
            print(f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
            
        # 保存评估结果
        with open("evaluation_report.txt", 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")

if __name__ == '__main__':
    main()