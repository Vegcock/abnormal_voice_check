import os
import glob
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

def load_audio(path, sr):
    """
    加载音频文件
    :param path:
    :param sr:
    :return:
    """
    y, _ = librosa.load(path, sr=sr, mono=True)

    return y

import numpy as np
import librosa
def compute_features(y, sr, n_mels=64, n_mfcc=13, hop_length=512, n_fft=2048):
    """
    增强版特征提取：
    - log-Mel spectrogram
    - delta & delta-delta
    - spectral contrast
    - MFCC + delta + delta-delta
    输出形状: (T, n_mels*3 + 7 + n_mfcc*3)
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6)  # (7, T)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length
    )  # (n_mfcc, T)

    T = min(
        log_mel.shape[1],
        contrast.shape[1],
        mfcc.shape[1],
    )
    log_mel = log_mel[:, :T]
    contrast = contrast[:, :T]
    mfcc = mfcc[:, :T]

    def normalize(x):
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        return (x - mean) / (std + 1e-8)

    log_mel = normalize(log_mel)
    contrast = normalize(contrast)
    mfcc = normalize(mfcc)

    feats = np.concatenate([
        log_mel,
        contrast,
        mfcc,
    ], axis=0)  # shape = (n_features, T)

    return feats.T.astype(np.float32)  # (T, n_features)

def compute_phase(y, n_fft=2048, hop_length=512):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    phase = np.angle(S)

    # unwrap to prevent jumps
    phase = np.unwrap(phase, axis=1)

    # temporal difference
    phase_diff = np.diff(phase, axis=1)
    phase_diff = np.pad(phase_diff, ((0, 0), (1, 0)), mode='constant')

    # normalize
    phase_diff = phase_diff / np.pi

    # (F, T) -> (T, F)
    return phase_diff.T.astype(np.float32)

def make_phase_windows(feats, phases, seq_len, hop_win):
    """
    feats: (T, feat_dim)
    phases: (T,)
    return: 列表，元素为 (feat_window, phase_window)
    """
    T = feats.shape[0]
    windows = []

    for start in range(0, T - seq_len, hop_win):
        end = start + seq_len
        w_feat = feats[start:end]
        w_phase = phases[start:end]
        windows.append((w_feat, w_phase))

    return windows

import re

def parse_section_from_path(path):
    m = re.search(r"section_(\d+)_", path)
    if m:
        return int(m.group(1))  # 从1开始
    return 0

class AudioWindowDataset(Dataset):
    def __init__(self, file_list, sr=22050, n_mels=64, n_mfcc=13, hop_length=512, n_fft=2048,
                 seq_len=64, hop_win=32, only_normal=True, train=False, device="fan"):
        self.train = train
        self.samples = []
        self.device = device
        section_names = set()
        device_names = set()
        for path in file_list:
            fname = os.path.basename(path).lower()
            parts = path.split(os.sep)
            if len(parts) >= 3:
                # 倒数第三个部分应该是设备名
                potential_device = parts[-3]
                device_names.add(potential_device.lower())
            else:
                # 如果路径格式不符，使用当前设备
                device_names.add(device.lower())
            if "section_" in fname:
                sec = fname.split("section_")[1].split("_")[0]
                section_names.add(sec)
        # 确保至少包含当前设备
        device_names.add(device.lower())
        if len(section_names) == 0:
            section_names.add("00")
        # 创建 LabelEncoder
        self.le = LabelEncoder()
        self.le.fit(sorted(list(device_names)))
        self.se_le = LabelEncoder()
        self.se_le.fit(sorted(list(section_names)))
        print(f"📌 设备类别: {list(self.le.classes_)}")
        print(f"📌 工况类别: {list(self.se_le.classes_)}")
        for path in tqdm(file_list, desc="数据集准备"):
            fname = os.path.basename(path).lower()
            if only_normal and ("normal" not in fname and "normal" not in path.lower()):
                continue
            # 1. 加载音频
            y = load_audio(path, sr=sr)
            # 2. 计算特征 (T, feat_dim)
            feats = compute_features(y, sr, n_mels=n_mels, hop_length=hop_length, n_mfcc=n_mfcc, n_fft=n_fft)
            # 3. 计算相位 (T,)
            phases = compute_phase(y, n_fft=n_fft, hop_length=hop_length)
            # 4. 切窗（窗口里包含：特征窗口 + 相位窗口）
            windows = make_phase_windows(feats, phases, seq_len, hop_win)
            # 从路径提取设备名并编码
            parts = path.split(os.sep)
            if len(parts) >= 3:
                file_device = parts[-3].lower()
            else:
                file_device = device.lower()

            if "section_" in fname:
                sec = fname.split("section_")[1].split("_")[0]
            else:
                sec = "00"
            section_label = self.se_le.transform([sec])[0]
            id_label = self.le.transform([file_device])[0]
            for w_idx, (wf, wp) in enumerate(windows):
                self.samples.append((wf, wp, path, w_idx, id_label,section_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wf, wp, path, widx, id_label, section_label = self.samples[idx]
        # 返回：特征窗口、相位窗口、文件路径、窗口编号、ID标签、工况标签
        return wf, wp, path, widx, id_label, section_label


class LinearPhaseEmbedding(nn.Module):
    def __init__(self, freq_bins, d_model):
        super().__init__()
        self.proj = nn.Linear(freq_bins, d_model)

    def forward(self, phase):
        # phase: (b, seq, freq_bins)
        return self.proj(phase)

# GWRP 全局加权排序池化
def gwrp_score(errors, r=0.5):
    e = np.array(errors, dtype=float)
    if e.size == 0:
        return 0.0
    ehat = np.sort(e)[::-1]  # descending
    I = len(ehat)
    idx = np.arange(I, dtype=float)
    weights = np.power(r, idx)
    Z = weights.sum()
    if Z == 0:
        Z = 1.0
    score = (weights * ehat).sum() / Z
    return float(score)

class TransAutoencoder(nn.Module):
    """
    Transformer-based Autoencoder with ID constraint
    and Phase Embedding from:
    "Transformer-based Autoencoder with ID Constraint for Unsupervised Anomalous Sound Detection"
    """

    def __init__(self,
                 feat_dim,
                 freq_bins,
                 d_model=128,
                 n_head=4,
                 num_layers=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 n_id=0,
                 n_section=0):

        super().__init__()

        self.feat_dim = feat_dim
        self.n_id = n_id
        self.n_section = n_section
        self.d_model = d_model

        # Feature projection
        self.input_proj = nn.Linear(feat_dim, d_model)

        # LPE: F -> d_model
        self.lpe = LinearPhaseEmbedding(freq_bins = freq_bins, d_model = d_model)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, feat_dim)

        # Center frame prediction head (CPE)
        self.center_pred = nn.Linear(d_model, feat_dim)

        # Section head
        if n_section > 0:
            self.section_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, n_section)
            )
        else:
            self.section_head = None

        # ID head
        if n_id > 0:
            self.id_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, n_id)
            )
        else:
            self.id_head = None

    def forward(self, x, phase=None):
        h = self.input_proj(x)

        if phase is not None:
            p_emb = self.lpe(phase)
            h = h + p_emb

        h = h.transpose(0, 1)
        out = self.encoder(h)
        out = out.transpose(0, 1)

        # 6) recon
        recon_seq = self.output_proj(out)

        # 7) pooled
        pooled = out.mean(dim=1)

        center_pred = self.center_pred(pooled)

        z = out.max(dim=1)[0]

        id_logits = self.id_head(z) if self.id_head else None
        section_logits = self.section_head(z) if self.section_head else None

        return recon_seq, id_logits, section_logits, center_pred, out

def collate_windows(batch):
    xs = torch.stack([torch.tensor(item[0],dtype=torch.float32) for item in batch], dim=0)
    phases = torch.stack([torch.tensor(item[1],dtype=torch.float32) for item in batch], dim=0)
    paths = [item[2] for item in batch]
    widxs = [item[3] for item in batch]
    id_labels = torch.tensor([item[4] for item in batch], dtype=torch.long)
    section_labels = torch.tensor([item[5] for item in batch], dtype=torch.long)
    return xs, phases, paths, widxs, id_labels, section_labels


def train(args):
    """
    args: 包含各种超参（epochs, batch_size, lr, alpha, top_k, threshold_percentile, gwrp_r 等）
    dataset: 一个 AudioWindowDataset（或类似）会在 samples 中包含 tuples:
             (window_feat (seq_len,D), phase_window (seq_len,) or None, file_path, window_idx, id_label)
             你需要在 DataLoader collate 中把这些打包成 (xb, phase_batch, paths, widxs, id_labels)
    device: torch.device
    返回: 保存的 checkpoint 路径, 阈值, LabelEncoder (若有)
    """
    base = os.path.join("data", args.device, "train", "*.wav")
    files = sorted(glob.glob(base))
    if len(files) == 0:
        raise RuntimeError(f"No train files found under {os.path.dirname(base)}")

    dataset = AudioWindowDataset(
        files, sr=args.sr, n_mels=args.n_mels, n_mfcc=args.n_mfcc,
        hop_length=args.hop_length, n_fft=args.n_fft,
        seq_len=args.seq_len, hop_win=args.hop_win,
        only_normal=True, train=True, device=args.device
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_windows
    )

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # 获取 LabelEncoder 和设备数量
    le = dataset.le
    se_le = dataset.se_le
    n_id = len(le.classes_) if le is not None else 0
    n_section = len(se_le.classes_) if se_le is not None else 0

    # model
    feat_dim = dataset.samples[0][0].shape[1]
    freq_bins = dataset.samples[0][1].shape[-1]
    model = TransAutoencoder(
        feat_dim,
        d_model=args.d_model,
        n_head=args.n_head,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        n_id=n_id,
        n_section=n_section,
        freq_bins=freq_bins
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.0))
    mse_loss = nn.MSELoss(reduction='mean')
    ce_loss = nn.CrossEntropyLoss()

    # 训练循环（中心帧预测 + ID 分类器 (若可用)）
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_id_loss = 0.0
        n_batches = 0

        loop = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)

        for batch in loop:
            xb, phase_batch, paths, widxs, id_labels, section_labels = batch
            xb = xb.to(device)  # (B, seq_len, feat_dim)
            phase_batch = phase_batch.to(device)  # (B, seq_len)
            id_labels = id_labels.to(device)  # (B,)
            section_labels = section_labels.to(device)

            optimizer.zero_grad()
            recon_seq, id_logits, section_logits, center_pred, per_frame_latent = model(
                xb, phase=phase_batch
            )

            Lr_full = mse_loss(recon_seq, xb)  # 全帧
            center_idx = xb.shape[1] // 2
            center_true = xb[:, center_idx, :]
            Lr_center = mse_loss(center_pred, center_true)  # 中心帧

            Lr = 0.5 * Lr_full + 0.5 * Lr_center

            # Lid = 0.0
            # if args.alpha > 0 and n_id > 1:
            #     Lid = ce_loss(id_logits, id_labels)
            #
            # Ls = 0.0
            # if args.beta > 0:
            #     Ls = ce_loss(section_logits, section_labels)

            loss = Lr # + args.alpha * Lid + args.beta * Ls

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_recon_loss += float(Lr.item())
            n_batches += 1

            # 更新进度条
            loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{Lr.item():.4f}',
                # 'id': f'{Lid.item():.4f}' if isinstance(Lid, torch.Tensor) else '0.0000'
            })

        avg_loss = total_loss / max(1, n_batches)
        avg_recon = total_recon_loss / max(1, n_batches)
        avg_id = total_id_loss / max(1, n_batches) if total_id_loss > 0 else 0.0

        print(f"\nEpoch {epoch}/{args.epochs} - Avg Loss: {avg_loss:.6f} | Recon: {avg_recon:.6f} | ID: {avg_id:.6f}")

    # 保存模型
    ckpt_path = f"checkpoint_{args.device}.pth"
    torch.save({
        'model_state': model.state_dict(),
        'args': vars(args),
        'label_encoder': le  # 保存 LabelEncoder
    }, ckpt_path)
    print("保存模型检查点:", ckpt_path)

    # 计算训练集上的重建误差分布，使用 GWRP 来得到每个窗口的异常分数，并以百分位设阈
    print("\n计算训练集重建误差分布以设置阈值...")
    model.eval()
    recon_scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="计算阈值"):
            xb, phase_batch, paths, widxs, id_labels, section_labels = batch
            xb = xb.to(device)
            phase_batch = phase_batch.to(device)

            recon_seq, id_logits, section_logits, center_pred, per_frame_latent = model(xb, phase=phase_batch)

            per_frame_mse = ((recon_seq - xb) ** 2).sum(dim=2).cpu().numpy()

            # 对每个sample用GWRP得到单个score
            for pf in per_frame_mse:
                score = gwrp_score(pf, r=args.gwrp_r)
                recon_scores.append(score)

    recon_scores = np.array(recon_scores)

    # 🔧 打印调试信息
    print(f"\n训练集误差统计:")
    print(f"   误差范围: [{recon_scores.min():.4f}, {recon_scores.max():.4f}]")
    print(f"   误差均值: {recon_scores.mean():.4f}")
    print(f"   误差标准差: {recon_scores.std():.4f}")
    print(f"   误差中位数: {np.median(recon_scores):.4f}")

    thresh = np.percentile(recon_scores, args.threshold_percentile)
    np.save(f"thresh_{args.device}.npy", np.array([thresh]))
    print(f"阈值 (percentile={args.threshold_percentile}%): {thresh:.6f}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.hist(recon_scores, bins=50)
    plt.axvline(thresh, color='r', linestyle='--', label=f'Threshold={thresh:.4f}')
    plt.title("Reconstruction Error Histogram")
    plt.xlabel("GWRP Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"reconstruction_error_hist_{args.device}.png")
    plt.close()
    print(f"保存误差直方图到 reconstruction_error_hist_{args.device}.png")

    return ckpt_path, thresh, le


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def test(args, ckpt_path, threshold_path):
    """
    测试流程：
      - 加载 checkpoint（若需要）
      - 遍历测试文件（dataset 中的 windows），对每个文件合并所有窗口的 frame-level errors，
        使用 GWRP 计算文件级异常分数，并结合 ID 分类器输出（若存在）做最终得分。
      - 输出并保存结果与评估指标
    """

    possible_paths = [
        os.path.join("data", args.device, args.test_folder, "*.wav"),
        os.path.join("data", args.device, "source_test", "*.wav"),
    ]

    test_files = []
    for path_pattern in possible_paths:
        files = sorted(glob.glob(path_pattern))
        if files:
            test_files = files
            print(f"从 {os.path.dirname(path_pattern)} 找到 {len(files)} 个测试文件")
            break

    if len(test_files) == 0:
        raise RuntimeError(f"未找到测试文件，已尝试路径: {possible_paths}")

    normal_files = [f for f in test_files if 'normal' in f.lower()]
    anomaly_files = [f for f in test_files if 'anomaly' in f.lower()]

    print(f"\n测试集统计:")
    print(f"   总文件数: {len(test_files)}")
    print(f"   正常文件: {len(normal_files)}")
    print(f"   异常文件: {len(anomaly_files)}")

    dataset = AudioWindowDataset(
        test_files, sr=args.sr, n_mels=args.n_mels, n_mfcc=args.n_mfcc,
        hop_length=args.hop_length, n_fft=args.n_fft,
        seq_len=args.seq_len, hop_win=args.hop_win,
        only_normal=False, train=False, device=args.device
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_windows
    )

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # 重建模型结构并加载权重
    ckpt = torch.load(ckpt_path, map_location="cpu")
    thresh = float(np.load(threshold_path)[0])
    print(f"加载的阈值: {thresh:.6f}")

    feat_dim = dataset.samples[0][0].shape[1]
    device_encoder = ckpt.get('le', getattr(dataset, 'le', None))
    section_encoder = ckpt.get('se_le', getattr(dataset, 'se_le', None))

    n_id = len(device_encoder.classes_) if device is not None else 0
    n_section = len(section_encoder.classes_) if section_encoder is not None else 0

    freq_bins = dataset.samples[0][1].shape[-1]
    model = TransAutoencoder(
        feat_dim, freq_bins=freq_bins, d_model=args.d_model, n_head=args.n_head,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, n_id=n_id, n_section=n_section
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    file_frame_errors = {}  # {file: {frame_idx: [errors]}}
    file_to_id_logits = {}  # {file: [logits]}

    print("\n开始推理测试集...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="测试中"):
            xb, phase_batch, paths, widxs, id_labels, section_labels = batch
            xb = xb.to(device)
            phase_batch = phase_batch.to(device)

            recon_seq, id_logits, section_logits, center_pred, per_frame_latent = model(xb, phase=phase_batch)

            per_frame_mse = ((recon_seq - xb) ** 2).sum(dim=2).cpu().numpy()  # (B, seq_len)

            # 将每个窗口的帧误差映射到全局帧索引
            for i in range(len(paths)):
                fn = paths[i]
                widx = widxs[i]
                pf = per_frame_mse[i]  # (seq_len,)

                # 初始化文件的帧误差字典
                if fn not in file_frame_errors:
                    file_frame_errors[fn] = {}

                # 将窗口中的每帧误差映射到全局帧索引
                for local_idx, err in enumerate(pf):
                    global_idx = widx * args.hop_win + local_idx
                    if global_idx not in file_frame_errors[fn]:
                        file_frame_errors[fn][global_idx] = []
                    file_frame_errors[fn][global_idx].append(float(err))

                # id logits optional (取平均作为文件级的 id 判定)
                if id_logits is not None:
                    logits = id_logits[i].cpu().numpy()
                    if fn not in file_to_id_logits:
                        file_to_id_logits[fn] = []
                    file_to_id_logits[fn].append(logits)

    # 平均重叠帧的误差并计算 GWRP
    file_to_errors = {}
    for fn, frame_dict in file_frame_errors.items():
        # 对每个全局帧索引的误差取平均（处理重叠）
        sorted_frames = sorted(frame_dict.items())
        avg_errors = [np.mean(errs) for idx, errs in sorted_frames]
        file_to_errors[fn] = avg_errors

    print("\n检查分数分布...")
    normal_errors = []
    anomaly_errors = []
    for fn, errs in list(file_to_errors.items())[:20]:  # 采样检查
        avg_err = np.mean(errs)
        if "normal" in fn.lower():
            normal_errors.append(avg_err)
        else:
            anomaly_errors.append(avg_err)

    # 计算每个文件的 GWRP score（不使用ID分类器）
    print("\n计算文件级异常分数...")
    results = []
    for fn, errs in file_to_errors.items():
        # 使用更激进的GWRP参数来放大差异
        gw = gwrp_score(errs, r=args.gwrp_r)  # r=0.2更关注最大误差

        final_score = gw
        id_term = 0.0

        # 判定异常（分数高于阈值）
        is_anom = final_score > thresh

        results.append((fn, final_score, is_anom, gw, id_term))

    #  添加分数分布检查
    all_scores = np.array([r[1] for r in results])
    normal_scores = np.array([r[1] for r in results if 'normal' in r[0].lower()])
    anomaly_scores = np.array([r[1] for r in results if 'anomaly' in r[0].lower()])

    print(f"\n测试集分数统计:")
    print(
        f"   全部: mean={all_scores.mean():.4f}, std={all_scores.std():.4f}, range=[{all_scores.min():.4f}, {all_scores.max():.4f}]")
    if len(normal_scores) > 0:
        print(f"   正常: mean={normal_scores.mean():.4f}, std={normal_scores.std():.4f}")
    if len(anomaly_scores) > 0:
        print(f"   异常: mean={anomaly_scores.mean():.4f}, std={anomaly_scores.std():.4f}")
    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        print(
            f"   差异: {anomaly_scores.mean() - normal_scores.mean():.4f} ({'异常>正常' if anomaly_scores.mean() > normal_scores.mean() else '⚠️ 异常<正常'})")

    # save results file
    time_str = datetime.now().strftime("%Y%m%d_%H%M")
    out_txt = f"results_{args.device}_{time_str}.txt"
    with open(out_txt, "w", encoding="utf-8") as fo:
        fo.write("文件路径\t标签\t综合分数\tGWRP分数\tID不确定度\n")
        for fn, score, is_anom, gw, id_term in results:
            lab = "异常" if is_anom else "正常"
            fo.write(f"{fn}\t{lab}\t{score:.6f}\t{gw:.6f}\t{id_term:.6f}\n")
    print(f"\n保存结果到 {out_txt}")

    # 评估指标
    y_true = []
    y_pred = []
    for fn, score, is_anom, gw, id_term in results:
        y_pred.append(1 if is_anom else 0)
        y_true.append(1 if "anomaly" in fn.lower() else 0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("测试结果")
    print("=" * 50)
    print(f"准确率 (Accuracy):  {acc:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"召回率 (Recall):    {rec:.4f}")
    print(f"F1 分数 (F1-Score): {f1:.4f}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="fan", help="设备子文件夹名")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--cuda", action="store_true", help="使用cuda进行训练")
    parser.add_argument("--sr", type=int, default=22050, help="每帧的采样量")
    parser.add_argument("--n_mels", type=int, default=128, help="Mel频带数（论文用128）")
    parser.add_argument("--n_mfcc", type=int, default=0, help="MFCC维度（设为0禁用，论文不用MFCC）")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512, help="默认值改为512以匹配相位计算")
    parser.add_argument("--seq_len", type=int, default=64, help="步长")
    parser.add_argument("--hop_win", type=int, default=32, help="窗口滑动步长")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--alpha", type=float, default=0.2, help="ID 损失的权重（0 表示禁用）")
    parser.add_argument("--beta", type=float, default=0.2, help="工况 损失的权重（0 表示禁用）")
    parser.add_argument("--id_weight", type=float, default=0.1, help="测试时 ID 不确定度的权重")
    parser.add_argument("--gwrp_r", type=float, default=0.2, help="GWRP 的 r 参数（更小=更关注最大误差）")
    parser.add_argument("--top_k", type=int, default=8, help="异常分数的前 k 个帧")
    parser.add_argument("--threshold_percentile", type=float, default=95.0, help="设定异常阈值的百分位数")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--threshold_path", type=str, default=None)
    parser.add_argument("--test_folder", type=str, default="source_test")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print(f"模式: {args.mode.upper()}")
    print(f"设备: {args.device}")
    print(f"配置: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"ID损失权重: {args.alpha}, 测试ID权重: {args.id_weight}")
    print("=" * 50 + "\n")

    if args.mode == "train":
        ckpt, thresh, le = train(args)
        print("\n训练完成!")
        print(f" 模型路径: {ckpt}")
        print(f" 阈值: {thresh:.6f}")
        print(f" 设备类别: {list(le.classes_) if le else 'None'}")
    else:
        if args.model_path is None or args.threshold_path is None:
            print("测试模式需要提供 --model_path 和 --threshold_path")
            print(
                "示例: python script.py --mode test --device fan --model_path checkpoint_fan.pth --threshold_path thresh_fan.npy")
        else:
            results = test(args, args.model_path, args.threshold_path)
            print("\n测试完成!")