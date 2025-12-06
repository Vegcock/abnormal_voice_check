"""
异常检测服务器
基于Flask的REST API，用于音频异常检测

启动方法:
    python server.py --device fan --port 5000

API端点:
    POST /predict - 上传音频文件进行异常检测
    GET /health - 健康检查
    GET /info - 获取模型信息
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import librosa
import os
import io
import argparse
import traceback
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 导入您的模型
from sectionTransformer import (
    AudioWindowDataset, TransAutoencoder,
    collate_windows, gwrp_score, compute_features, compute_phase
)

# 导入CNN模型和特征提取
from supervised_classifier import extract_discriminative_features

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
model = None  # Transformer模型
cnn_model = None  # CNN模型
device = None
args_config = None
threshold = None
section_encoder = None
id_encoder = None
cnn_scaler = None  # CNN模型的标准化器
current_model_type = None  # 'transformer' 或 'cnn'

# 配置
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def allowed_file(filename):
    """检查文件扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model(device_name, model_path, threshold_path):
    """加载模型"""
    global model, device, args_config, threshold, section_encoder, id_encoder

    print(f"🔄 加载模型: {model_path}")

    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    args_config = argparse.Namespace(**checkpoint.get('args', {}))
    threshold = np.load(threshold_path)[0] if os.path.exists(threshold_path) else None

    # 获取编码器
    section_encoder = checkpoint.get('section_encoder', None)
    id_encoder = checkpoint.get('id_encoder', None)

    n_id = checkpoint.get('n_id', 0)
    n_section = checkpoint.get('n_section', 0)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 🔧 从checkpoint推断特征维度
    # 查找input_proj.weight的形状来确定feat_dim
    model_state = checkpoint['model_state']
    input_proj_weight = model_state.get('input_proj.weight')

    if input_proj_weight is not None:
        feat_dim = input_proj_weight.shape[1]  # (d_model, feat_dim) -> 取feat_dim
        print(f"   从checkpoint推断feat_dim: {feat_dim}")
    else:
        # 默认值
        n_mels = getattr(args_config, 'n_mels', 128)
        feat_dim = n_mels * 3 + 7
        print(f"   使用默认feat_dim: {feat_dim}")

    # 🔧 从checkpoint推断freq_bins
    lpe_proj_weight = model_state.get('lpe.proj.weight')
    if lpe_proj_weight is not None:
        freq_bins = lpe_proj_weight.shape[1]  # (d_model, freq_bins) -> 取freq_bins
        print(f"   从checkpoint推断freq_bins: {freq_bins}")
    else:
        freq_bins = getattr(args_config, 'n_mels', 128)
        print(f"   使用默认freq_bins: {freq_bins}")

    # 创建模型（使用新的结构）
    model = TransAutoencoder(
        feat_dim=feat_dim,
        freq_bins=freq_bins,
        d_model=getattr(args_config, 'd_model', 128),
        n_head=getattr(args_config, 'n_head', 4),
        num_layers=getattr(args_config, 'num_layers', 3),
        dim_feedforward=getattr(args_config, 'dim_feedforward', 256),
        dropout=getattr(args_config, 'dropout', 0.1),
        n_id=n_id,
        n_section=n_section
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()

    print(f"✅ 模型加载完成")
    print(f"   设备: {device}")
    print(f"   特征维度: {feat_dim}")
    print(f"   频率bins: {freq_bins}")
    print(f"   阈值: {threshold:.4f}")
    print(f"   ID类别: {n_id}")
    print(f"   Section类别: {n_section}")

    return True


class CNNClassifier(nn.Module):
    """CNN分类器模型（与supervised_classifier.py中的定义一致）"""
    def __init__(self, input_dim, num_classes=2):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def load_cnn_model(device_name, model_path):
    """加载CNN模型"""
    global cnn_model, cnn_scaler, device
    
    print(f"🔄 加载CNN模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    input_dim = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']
    
    # 重建标准化器
    cnn_scaler = StandardScaler()
    cnn_scaler.mean_ = checkpoint['scaler_mean']
    cnn_scaler.scale_ = checkpoint['scaler_scale']
    
    # 创建模型并加载权重
    device_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = CNNClassifier(input_dim=input_dim, num_classes=num_classes).to(device_torch)
    cnn_model.load_state_dict(checkpoint['model_state'])
    cnn_model.eval()
    
    # 如果device还未设置，则设置为CNN模型的device
    if device is None:
        device = device_torch
    
    print(f"✅ CNN模型加载完成")
    print(f"   设备: {device_torch}")
    print(f"   输入维度: {input_dim}")
    print(f"   测试F1: {checkpoint.get('test_f1', 'N/A')}")
    
    return True


def analyze_audio_features(y, sr):
    """分析音频的声纹特征，返回详细的特征信息"""
    features = {}
    
    # 1. MFCC特征
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features['mfcc'] = {
        'mean': np.mean(mfcc, axis=1).tolist(),
        'std': np.std(mfcc, axis=1).tolist(),
        'skew': stats.skew(mfcc, axis=1).tolist(),
        'kurtosis': stats.kurtosis(mfcc, axis=1).tolist(),
        'anomaly_score': float(np.mean(np.abs(np.mean(mfcc, axis=1) - np.mean(mfcc))))  # 简化的异常分数
    }
    
    # 2. Mel频谱特征
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features['mel'] = {
        'mean': np.mean(mel_db, axis=1).tolist(),
        'std': np.std(mel_db, axis=1).tolist(),
        'max': np.max(mel_db, axis=1).tolist(),
        'min': np.min(mel_db, axis=1).tolist(),
        'anomaly_score': float(np.std(np.mean(mel_db, axis=1)))  # 使用标准差作为异常指标
    }
    
    # 3. 频谱特征
    S = np.abs(librosa.stft(y))
    features['spectral'] = {
        'centroid': {
            'mean': float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))),
            'std': float(np.std(librosa.feature.spectral_centroid(S=S, sr=sr)))
        },
        'rolloff': {
            'mean': float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))),
            'std': float(np.std(librosa.feature.spectral_rolloff(S=S, sr=sr)))
        },
        'bandwidth': {
            'mean': float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))),
            'std': float(np.std(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
        },
        'flatness': {
            'mean': float(np.mean(librosa.feature.spectral_flatness(S=S))),
            'std': float(np.std(librosa.feature.spectral_flatness(S=S)))
        },
        'contrast': {
            'mean': np.mean(librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6), axis=1).tolist(),
            'std': np.std(librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6), axis=1).tolist()
        }
    }
    
    # 4. 能量特征
    features['energy'] = {
        'rms': float(np.sqrt(np.mean(y ** 2))),
        'zcr_mean': float(np.mean(librosa.feature.zero_crossing_rate(y))),
        'zcr_std': float(np.std(librosa.feature.zero_crossing_rate(y)))
    }
    
    # 5. 谐波-冲击特征
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    features['harmonic_percussive'] = {
        'harmonic_energy': float(np.sqrt(np.mean(y_harmonic ** 2))),
        'percussive_energy': float(np.sqrt(np.mean(y_percussive ** 2))),
        'ratio': float(np.sqrt(np.mean(y_harmonic ** 2)) / (np.sqrt(np.mean(y_percussive ** 2)) + 1e-6))
    }
    
    return features


def predict_audio_cnn(audio_path):
    """使用CNN模型预测音频文件，返回详细的特征分析"""
    global cnn_model, cnn_scaler, device
    
    if cnn_model is None:
        raise RuntimeError("CNN模型未加载")
    
    # 获取设备（如果device未设置，使用模型所在的设备）
    device_torch = next(cnn_model.parameters()).device if device is None else device
    
    # 1. 加载音频
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # 2. 提取特征
    feats = extract_discriminative_features(y, sr)
    
    # 3. 标准化
    feats_scaled = cnn_scaler.transform(feats.reshape(1, -1))
    
    # 4. 预测
    feats_tensor = torch.FloatTensor(feats_scaled).to(device_torch)
    cnn_model.eval()
    with torch.no_grad():
        outputs = cnn_model(feats_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # 5. 获取预测结果
    is_anomaly = bool(predicted.item() == 1)
    anomaly_prob = float(probabilities[0][1].item())
    normal_prob = float(probabilities[0][0].item())
    confidence = float(max(anomaly_prob, normal_prob))
    
    # 6. 分析声纹特征
    feature_analysis = analyze_audio_features(y, sr)
    
    # 7. 计算综合异常分数（基于特征分析）
    # 使用多个特征的异常指标综合计算
    anomaly_scores = {
        'mfcc': feature_analysis['mfcc']['anomaly_score'],
        'mel': feature_analysis['mel']['anomaly_score'],
        'spectral_flatness': feature_analysis['spectral']['flatness']['std'],
        'spectral_bandwidth': feature_analysis['spectral']['bandwidth']['std'],
        'overall': anomaly_prob  # CNN模型的预测概率作为综合分数
    }
    
    return {
        'is_anomaly': is_anomaly,
        'anomaly_probability': anomaly_prob,
        'normal_probability': normal_prob,
        'confidence': confidence,
        'anomaly_scores': anomaly_scores,
        'feature_analysis': feature_analysis,
        'audio_duration': float(len(y) / sr),
        'sample_rate': int(sr)
    }


def predict_audio(audio_path):
    """预测单个音频文件"""
    global model, device, args_config, threshold

    if model is None:
        raise RuntimeError("模型未加载")

    # 1. 加载音频
    y, sr = librosa.load(audio_path, sr=getattr(args_config, 'sr', 22050), mono=True)

    # 2. 提取特征
    feats = compute_features(
        y, sr,
        n_mels=getattr(args_config, 'n_mels', 128),
        hop_length=getattr(args_config, 'hop_length', 512),
        n_mfcc=getattr(args_config, 'n_mfcc', 0),
        n_fft=getattr(args_config, 'n_fft', 2048)
    )

    # 3. 计算相位
    phases = compute_phase(
        y,
        n_fft=getattr(args_config, 'n_fft', 2048),
        hop_length=getattr(args_config, 'hop_length', 512)
    )

    # 4. 切窗
    seq_len = getattr(args_config, 'seq_len', 32)
    hop_win = getattr(args_config, 'hop_win', 16)

    T = min(feats.shape[0], phases.shape[0])
    feats = feats[:T]
    phases = phases[:T]

    windows = []
    for start in range(0, T - seq_len, hop_win):
        end = start + seq_len
        windows.append((feats[start:end], phases[start:end]))

    if len(windows) == 0:
        raise ValueError(f"音频太短，至少需要 {seq_len} 帧")

    # 5. 批量预测
    frame_errors = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i+batch_size]

            # 准备batch
            xs = torch.stack([torch.FloatTensor(w[0]) for w in batch_windows], dim=0).to(device)
            phases_batch = torch.stack([torch.FloatTensor(w[1]) for w in batch_windows], dim=0).to(device)

            # 前向传播（新模型的forward不返回per_frame_latent）
            recon_seq, _, _, _, _ = model(xs, phase=phases_batch)

            # 计算误差
            per_frame_mse = ((recon_seq - xs) ** 2).sum(dim=2).cpu().numpy()

            for pf in per_frame_mse:
                frame_errors.extend(pf.tolist())

    # 6. 计算GWRP分数
    gwrp_r = getattr(args_config, 'gwrp_r', 0.5)
    anomaly_score = gwrp_score(frame_errors, r=gwrp_r)

    # 7. 判定
    is_anomaly = anomaly_score > threshold if threshold is not None else False
    confidence = min(abs(anomaly_score - threshold) / threshold, 1.0) if threshold else 0.5

    return {
        'anomaly_score': float(anomaly_score),
        'threshold': float(threshold) if threshold else None,
        'is_anomaly': bool(is_anomaly),
        'confidence': float(confidence),
        'num_frames': len(frame_errors),
        'audio_duration': float(len(y) / sr)
    }

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/info', methods=['GET'])
@app.route('/info/<model_type>', methods=['GET'])
def model_info(model_type=None):
    """获取模型信息，支持通过URL参数选择模型"""
    # 获取模型类型
    if model_type is None:
        model_type = request.args.get('model', 'transformer')
    model_type = model_type.lower()
    
    if model_type == 'transformer':
        if model is None:
            return jsonify({'error': 'Transformer模型未加载'}), 503
        
        return jsonify({
            'model_type': 'TransformerAutoencoder',
            'device': str(device),
            'threshold': float(threshold) if threshold else None,
            'config': {
                'seq_len': getattr(args_config, 'seq_len', None),
                'n_mels': getattr(args_config, 'n_mels', None),
                'd_model': getattr(args_config, 'd_model', None),
                'num_layers': getattr(args_config, 'num_layers', None)
            },
            'sections': list(section_encoder.classes_) if section_encoder else [],
            'device_ids': list(id_encoder.classes_) if id_encoder else []
        })
    elif model_type == 'cnn':
        if cnn_model is None:
            return jsonify({'error': 'CNN模型未加载'}), 503
        
        # 获取CNN模型信息（需要从checkpoint中读取）
        return jsonify({
            'model_type': 'CNNClassifier',
            'device': str(device),
            'status': 'loaded'
        })
    else:
        return jsonify({'error': f'不支持的模型类型: {model_type}'}), 400

@app.route('/predict', methods=['POST'])
@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type=None):
    """预测接口，支持通过URL参数或查询参数选择模型
    
    支持的模型类型:
    - transformer: Transformer自编码器模型（默认）
    - cnn: CNN分类器模型
    
    使用方式:
    - POST /predict?model=transformer
    - POST /predict?model=cnn
    - POST /predict/transformer
    - POST /predict/cnn
    """
    try:
        # 获取模型类型（优先从URL路径，其次从查询参数，最后从表单数据）
        if model_type is None:
            model_type = request.args.get('model', request.form.get('model', 'transformer'))
        
        model_type = model_type.lower()
        
        if model_type not in ['transformer', 'cnn']:
            return jsonify({
                'error': f'不支持的模型类型: {model_type}，支持: transformer, cnn'
            }), 400

        # 检查模型是否加载
        if model_type == 'transformer':
            if model is None:
                return jsonify({'error': 'Transformer模型未加载'}), 503
        elif model_type == 'cnn':
            if cnn_model is None:
                return jsonify({'error': 'CNN模型未加载'}), 503

        # 检查文件
        if 'file' not in request.files:
            return jsonify({'error': '未找到文件'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': f'不支持的文件格式，支持: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # 保存临时文件
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{datetime.now().timestamp()}_{filename}")
        file.save(temp_path)

        try:
            # 根据模型类型进行预测
            if model_type == 'transformer':
                result = predict_audio(temp_path)
            else:  # cnn
                result = predict_audio_cnn(temp_path)

            # 添加元数据
            result['filename'] = filename
            result['timestamp'] = datetime.now().isoformat()
            result['status'] = 'success'
            result['model_type'] = model_type

            return jsonify(result), 200

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        print(f"❌ 预测错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测接口"""
    try:
        if model is None:
            return jsonify({'error': '模型未加载'}), 503

        files = request.files.getlist('files')

        if not files:
            return jsonify({'error': '未找到文件'}), 400

        results = []

        for file in files:
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': '不支持的文件格式'
                })
                continue

            filename = secure_filename(file.filename)
            # 使用跨平台的临时目录
            temp_dir = tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)  # 确保目录存在
            temp_path = os.path.join(temp_dir, f"{datetime.now().timestamp()}_{filename}")

            try:
                file.save(temp_path)
                result = predict_audio(temp_path)
                result['filename'] = filename
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                results.append({
                    'filename': filename,
                    'status': 'error',
                    'error': str(e)
                })
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        return jsonify({
            'results': results,
            'total': len(files),
            'success': sum(1 for r in results if r['status'] == 'success'),
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """文件过大"""
    return jsonify({'error': '文件过大，最大50MB'}), 413

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='异常检测服务器')
    parser.add_argument('--device', type=str, default='fan', help='设备名称')
    parser.add_argument('--model_path', type=str, default=None, help='Transformer模型路径')
    parser.add_argument('--threshold_path', type=str, default=None, help='阈值路径')
    parser.add_argument('--cnn_model_path', type=str, default=None, help='CNN模型路径')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='主机地址')
    parser.add_argument('--port', type=int, default=5000, help='端口号')
    parser.add_argument('--debug', action='store_true', help='调试模式')

    args = parser.parse_args()

    # 默认路径
    if args.model_path is None:
        args.model_path = f'checkpoint_{args.device}.pth'
    if args.threshold_path is None:
        args.threshold_path = f'thresh_{args.device}.npy'
    if args.cnn_model_path is None:
        args.cnn_model_path = f'ckpt_cnn_{args.device}.pth'

    # 加载Transformer模型
    try:
        load_model(args.device, args.model_path, args.threshold_path)
        current_model_type = 'transformer'
    except Exception as e:
        print(f"⚠️  Transformer模型加载失败: {e}")
        print("提示: Transformer模型将不可用")
        current_model_type = None

    # 加载CNN模型
    try:
        load_cnn_model(args.device, args.cnn_model_path)
        if current_model_type is None:
            current_model_type = 'cnn'
    except Exception as e:
        print(f"⚠️  CNN模型加载失败: {e}")
        print("提示: CNN模型将不可用")

    if model is None and cnn_model is None:
        print("❌ 没有可用的模型，服务器无法启动")
        exit(1)

    # 配置文件上传大小
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

    print("\n" + "="*70)
    print(f"🚀 异常检测服务器启动")
    print("="*70)
    print(f"📍 地址: http://{args.host}:{args.port}")
    print(f"📱 设备: {args.device}")
    print(f"💡 API端点:")
    print(f"   POST /predict              - 单文件预测（默认transformer）")
    print(f"   POST /predict/transformer - 使用Transformer模型预测")
    print(f"   POST /predict/cnn         - 使用CNN模型预测（包含特征分析）")
    print(f"   POST /predict?model=cnn   - 通过查询参数选择模型")
    print(f"   POST /batch_predict        - 批量预测")
    print(f"   GET  /health               - 健康检查")
    print(f"   GET  /info                 - 模型信息（默认transformer）")
    print(f"   GET  /info/transformer     - Transformer模型信息")
    print(f"   GET  /info/cnn             - CNN模型信息")
    print("="*70 + "\n")

    # 启动服务器
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )