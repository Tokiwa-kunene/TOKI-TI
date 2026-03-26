# ========================================
# 文件3: utils.py - 工具函数
# ========================================
"""
工具函数模块
"""
import torch
import numpy as np
import random
import os
import json
from datetime import datetime


def set_seed(seed):
    """
    设置随机种子以确保可复现性

    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config, save_path):
    """
    保存配置到JSON文件

    Args:
        config: 配置对象
        save_path: 保存路径
    """
    config_dict = {
        key: value for key, value in vars(config).items()
        if not key.startswith('_') and not callable(value)
    }
    # 转换torch.device为字符串
    if 'DEVICE' in config_dict:
        config_dict['DEVICE'] = str(config_dict['DEVICE'])

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    print(f"配置已保存到: {save_path}")


def get_timestamp():
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_model_info(model):
    """
    打印模型信息

    Args:
        model: PyTorch模型
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 50)
    print("模型信息")
    print("=" * 50)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    print("=" * 50 + "\n")
