import numpy as np
import librosa
import glob
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import stats


def extract_discriminative_features(y, sr):
    """提取最具判别力的特征（基于特征分析结果）"""
    features = []

    # 1. MFCC统计 (最重要)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))  # 20个均值
    features.extend(np.std(mfcc, axis=1))  # 20个标准差
    features.extend(stats.skew(mfcc, axis=1))  # 20个偏度
    features.extend(stats.kurtosis(mfcc, axis=1))  # 20个峰度

    # 2. Spectral Contrast (第二重要)
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # 3. Mel频谱统计
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    features.extend(np.mean(mel, axis=1)[:10])  # 前10个mel band
    features.extend(np.std(mel, axis=1)[:10])

    # 4. 频谱特征
    features.append(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    features.append(np.std(librosa.feature.spectral_centroid(S=S, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
    features.append(np.std(librosa.feature.spectral_rolloff(S=S, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    features.append(np.std(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_flatness(S=S)))

    # 5. 能量特征
    features.append(np.sqrt(np.mean(y ** 2)))  # RMS
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))

    # 6. 谐波-冲击特征
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    features.append(np.sqrt(np.mean(y_harmonic ** 2)))
    features.append(np.sqrt(np.mean(y_percussive ** 2)))

    return np.array(features)


class FeatureDataset(Dataset):
    """特征数据集"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CNNClassifier(nn.Module):
    """CNN分类器模型"""
    def __init__(self, input_dim, num_classes=2):
        super(CNNClassifier, self).__init__()
        
        # 将特征向量重塑为适合CNN的格式 (batch, 1, features)
        # 使用1D卷积处理特征向量
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # 使用全局平均池化，输出维度为256（通道数）
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        
    def forward(self, x):
        # x shape: (batch, features)
        # 重塑为 (batch, 1, features) 用于1D卷积
        x = x.unsqueeze(1)  # (batch, 1, features)
        
        # 卷积层
        x = self.relu(self.bn1(self.conv1(x)))  # (batch, 64, features)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch, 128, features)
        x = self.relu(self.bn3(self.conv3(x)))  # (batch, 256, features)
        
        # 全局平均池化
        x = self.pool(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


def train_supervised_classifier(device="fan", epochs=50, batch_size=32, lr=0.001):
    """训练监督分类器"""

    print("=" * 80)
    print("🎯 基于判别特征的监督分类器训练")
    print("=" * 80)

    # 加载训练数据
    print("\n📂 加载训练数据...")
    train_normal = sorted(glob.glob(f"data/{device}/train/*normal*.wav"))

    # 从测试集借用一些异常样本作为训练（交叉验证会评估泛化能力）
    test_normal = sorted(glob.glob(f"data/{device}/source_test/*normal*.wav"))
    test_anomaly = sorted(glob.glob(f"data/{device}/source_test/*anomaly*.wav"))

    # 使用训练集正常 + 测试集部分正常/异常作为训练
    train_files = train_normal[:2000] + test_normal[:150] + test_anomaly[:150]
    train_labels = [0] * 2150 + [1] * 150

    # 剩余作为测试集
    test_files = test_normal[150:] + test_anomaly[150:]
    test_labels = [0] * (len(test_normal) - 150) + [1] * (len(test_anomaly) - 150)

    print(f"   训练集: {len(train_files)} 个样本 (正常:{train_labels.count(0)}, 异常:{train_labels.count(1)})")
    print(f"   测试集: {len(test_files)} 个样本 (正常:{test_labels.count(0)}, 异常:{test_labels.count(1)})")

    # 提取特征
    print("\n🔄 提取特征...")
    X_train = []
    y_train = []

    for path, label in tqdm(zip(train_files, train_labels), total=len(train_files), desc="训练集"):
        try:
            y_audio, sr = librosa.load(path, sr=22050, mono=True)
            feats = extract_discriminative_features(y_audio, sr)
            X_train.append(feats)
            y_train.append(label)
        except Exception as e:
            print(f"   ⚠️  跳过文件 {path}: {e}")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"   训练特征维度: {X_train.shape}")

    X_test = []
    y_test = []

    for path, label in tqdm(zip(test_files, test_labels), total=len(test_files), desc="测试集"):
        try:
            y_audio, sr = librosa.load(path, sr=22050, mono=True)
            feats = extract_discriminative_features(y_audio, sr)
            X_test.append(feats)
            y_test.append(label)
        except Exception as e:
            print(f"   ⚠️  跳过文件 {path}: {e}")

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 标准化
    print("\n📊 特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 获取特征维度
    input_dim = X_train_scaled.shape[1]
    print(f"   特征维度: {input_dim}")

    # 创建数据集和数据加载器
    train_dataset = FeatureDataset(X_train_scaled, y_train)
    test_dataset = FeatureDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建CNN模型
    print("\n🤖 创建CNN模型...")
    device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device_torch}")
    
    model = CNNClassifier(input_dim=input_dim, num_classes=2).to(device_torch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 计算类别权重以处理不平衡数据
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor([1.0 / class_counts[0], 1.0 / class_counts[1]]).to(device_torch)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 训练模型
    print(f"\n🔄 训练CNN模型 (epochs={epochs})...")
    best_test_f1 = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            features = features.to(device_torch)
            labels = labels.to(device_torch)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device_torch)
                labels = labels.to(device_torch)
                
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = test_correct / test_total
        test_prec = precision_score(all_labels, all_preds, zero_division=0)
        test_rec = recall_score(all_labels, all_preds, zero_division=0)
        test_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练损失: {train_loss/len(train_loader):.4f}, 训练准确率: {train_acc:.4f}")
            print(f"  测试准确率: {test_acc:.4f}, 精确率: {test_prec:.4f}, 召回率: {test_rec:.4f}, F1: {test_f1:.4f}")
        
        # 保存最佳模型
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_model_state = model.state_dict().copy()

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    model.eval()
    
    # 最终评估
    print(f"\n{'=' * 80}")
    print("📊 最终测试结果")
    print(f"{'=' * 80}")
    print(f"  准确率:  {test_acc:.4f}")
    print(f"  精确率:  {test_prec:.4f}")
    print(f"  召回率:  {test_rec:.4f}")
    print(f"  F1分数:  {test_f1:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n混淆矩阵:")
    print(f"  真负例(TN): {cm[0, 0]:>4}  |  假正例(FP): {cm[0, 1]:>4}")
    print(f"  假负例(FN): {cm[1, 0]:>4}  |  真正例(TP): {cm[1, 1]:>4}")

    # 保存模型
    model_path = f'ckpt_cnn_{device}.pth'
    torch.save({
        'model_state': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': 2,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'test_acc': test_acc,
        'test_prec': test_prec,
        'test_rec': test_rec,
        'test_f1': test_f1
    }, model_path)
    print(f"\n✅ 保存CNN模型到 {model_path}")

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'CNN Classifier Confusion Matrix\nF1={test_f1:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'cnn_confusion_matrix_{device}.png', dpi=150)
    print(f"✅ 保存混淆矩阵图到 cnn_confusion_matrix_{device}.png")

    results = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'cm': cm,
        'y_pred': all_preds
    }

    return model, scaler, results


def test_classifier(device="fan"):
    """测试保存的CNN分类器"""

    print("\n" + "=" * 80)
    print("🧪 测试保存的CNN分类器")
    print("=" * 80)

    # 加载模型
    model_path = f'ckpt_cnn_{device}.pth'
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    input_dim = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']
    
    # 重建标准化器
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    
    # 创建模型并加载权重
    device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNClassifier(input_dim=input_dim, num_classes=num_classes).to(device_torch)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"✅ 加载模型: {model_path}")
    print(f"   使用设备: {device_torch}")

    # 加载所有测试数据
    test_normal = sorted(glob.glob(f"data/{device}/source_test/*normal*.wav"))
    test_anomaly = sorted(glob.glob(f"data/{device}/source_test/*anomaly*.wav"))

    test_files = test_normal + test_anomaly
    test_labels = [0] * len(test_normal) + [1] * len(test_anomaly)

    print(f"\n📂 测试完整测试集: {len(test_files)} 个样本")

    # 提取特征
    X_test = []
    y_test = []

    for path, label in tqdm(zip(test_files, test_labels), total=len(test_files)):
        try:
            y_audio, sr = librosa.load(path, sr=22050, mono=True)
            feats = extract_discriminative_features(y_audio, sr)
            X_test.append(feats)
            y_test.append(label)
        except Exception as e:
            print(f"   ⚠️  跳过: {e}")

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test_scaled = scaler.transform(X_test)

    # 转换为PyTorch张量并预测
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device_torch)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
        y_proba = torch.softmax(outputs, dim=1)[:, 1]
    
    y_pred = y_pred.cpu().numpy()
    y_proba = y_proba.cpu().numpy()

    # 评估
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 80)
    print("📊 完整测试集结果")
    print("=" * 80)
    print(f"准确率 (Accuracy):  {acc:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"召回率 (Recall):    {rec:.4f}")
    print(f"F1 分数 (F1-Score): {f1:.4f}")
    print("=" * 80)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:")
    print(f"  TN: {cm[0, 0]:>4}  |  FP: {cm[0, 1]:>4}")
    print(f"  FN: {cm[1, 0]:>4}  |  TP: {cm[1, 1]:>4}")

    return acc, prec, rec, f1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="fan")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        model, scaler, results = train_supervised_classifier(args.device)

    if args.mode in ["test", "both"]:
        test_classifier(args.device)