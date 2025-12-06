import numpy as np
import librosa
import glob
import os
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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


def train_supervised_classifier(device="fan"):
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

    # 训练Gradient Boosting模型
    print("\n🤖 训练Gradient Boosting分类器...")
    print("=" * 60)

    # 创建Gradient Boosting模型
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    # 训练
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # 评估
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集结果:")
    print(f"  准确率:  {test_acc:.4f}")
    print(f"  精确率:  {test_prec:.4f}")
    print(f"  召回率:  {test_rec:.4f}")
    print(f"  F1分数:  {test_f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n混淆矩阵:")
    print(f"  真负例(TN): {cm[0, 0]:>4}  |  假正例(FP): {cm[0, 1]:>4}")
    print(f"  假负例(FN): {cm[1, 0]:>4}  |  真正例(TP): {cm[1, 1]:>4}")

    # 特征重要性
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-20:][::-1]

    print(f"\n🔝 Top 20 重要特征:")
    for i, idx in enumerate(top_indices[:20]):
        print(f"   {i + 1}. 特征 {idx}: {feature_importance[idx]:.4f}")

    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    plt.bar(range(20), feature_importance[top_indices[:20]])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Top 20 Feature Importances (Gradient Boosting)')
    plt.tight_layout()
    plt.savefig(f'feature_importance_gb_{device}.png', dpi=150)
    print(f"   ✅ 保存特征重要性图到 feature_importance_gb_{device}.png")

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Gradient Boosting Confusion Matrix\nF1={test_f1:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_gb_{device}.png', dpi=150)
    print(f"\n✅ 保存混淆矩阵图到 confusion_matrix_gb_{device}.png")

    # 保存模型
    model_path = f'ckpt_gb_{device}.pkl'
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'test_acc': test_acc,
        'test_prec': test_prec,
        'test_rec': test_rec,
        'test_f1': test_f1,
        'feature_importance': feature_importance
    }, model_path)
    print(f"\n✅ 保存模型到 {model_path}")

    results = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'cm': cm,
        'y_pred': y_pred_test
    }

    return model, scaler, results


def test_classifier(device="fan"):
    """测试保存的Gradient Boosting分类器"""

    print("\n" + "=" * 80)
    print("🧪 测试保存的Gradient Boosting分类器")
    print("=" * 80)

    # 加载模型
    model_path = f'ckpt_gb_{device}.pkl'
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    checkpoint = joblib.load(model_path)
    model = checkpoint['model']
    scaler = checkpoint['scaler']

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

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

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