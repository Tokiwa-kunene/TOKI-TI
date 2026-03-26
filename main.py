"""
主程序入口 - SimCSE版本
运行方式: python main.py

SimCSE-ABSA: SimCSE-based Cross-lingual ABSA
核心改进：
1. SimCSE (Dropout-based Unsupervised Contrastive Learning)
2. 无需翻译数据，避免翻译噪声
3. Attention Pooling 提取情感特征
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import warnings

warnings.filterwarnings('ignore')

# 导入自定义模块
from config import Config
from utils import set_seed, save_config, print_model_info, get_timestamp
from data_preprocessing import DataPreprocessor
from DatasetPY.dataset_version2 import CrossLingualDataset
from model.model_version2 import XLMSentimentModel
from trainer.trainer_version2 import Trainer


def main():
    """主函数"""
    # 创建配置实例以便读取语言设置
    config = Config()

    train_lang = config.get_lang_name(config.TRAIN_LANG)
    valid_lang = config.get_lang_name(config.VALID_LANG)
    test_lang = config.get_lang_name(config.TEST_LANG)


    print("\n" + "=" * 70)
    print("SimCSE-ABSA: SimCSE-based Cross-lingual ABSA")
    print(f"跨语言情感分析项目 - {train_lang} -> {test_lang} 情感迁移")
    print(f"训练集：{train_lang}")
    print(f"验证集：{valid_lang}")
    print(f"测试集：{test_lang}")
    print("=" * 70)
    # ... (后续保持不变) ...
    print("\n核心改进:")
    print("  [1] SimCSE - Dropout-based Contrastive Learning")
    print("  [2] 无需翻译数据 - 避免翻译噪声")
    print("  [3] Attention Pooling - 捕捉情感关键词")
    print("=" * 70)

    config.create_dirs()

    # 设置随机种子
    set_seed(config.SEED)
    print(f"\n随机种子: {config.SEED}")
    print(f"使用设备: {config.DEVICE}")

    # 打印配置信息
    config.print_config()

    # 打印性能优化建议
    tips = config.get_performance_tips()
    if tips:
        print("=" * 70)
        print("💡 性能优化建议:")
        print("=" * 70)
        for tip in tips:
            print(f"  {tip}")
        print("=" * 70 + "\n")

    # 保存配置
    timestamp = get_timestamp()
    config_save_path = f"{config.LOG_DIR}/config_{timestamp}.json"
    save_config(config, config_save_path)

    # ========== 步骤1: 数据预处理 ==========
    print("\n" + "=" * 70)
    print("步骤1: 数据预处理 (SimCSE模式)")
    print("=" * 70)

    preprocessor = DataPreprocessor(config)
    train_data, valid_data, test_data = preprocessor.prepare_datasets()

    # ========== 步骤2: 加载Tokenizer ==========
    print("=" * 70)
    print("步骤2: 加载Tokenizer")
    print("=" * 70)

    # AutoTokenizer 会读取模型文件夹里的 config.json，自动识别这是 mBERT 还是 XLM-R
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    print("✓ Tokenizer加载完成\n")

    # ========== 步骤3: 创建数据集和数据加载器 ==========
    print("=" * 70)
    print("步骤3: 创建数据加载器")
    print("=" * 70)

    # 训练集
    train_dataset = CrossLingualDataset(
        train_data,         #原始数据
        tokenizer,          #token用的工具
        config.MAX_LENGTH   #设定最大长度
    )

    # DataLoader配置
    dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,                #一次打包的句子数量
        'num_workers': config.NUM_WORKERS,              #进程数
        'pin_memory': config.PIN_MEMORY and config.DEVICE.type == 'cuda',
    }

    '''
    决定是否在数据加载器（DataLoader）中开启工作进程持久化功能。
    在 PyTorch 的默认机制下，每一轮训练（Epoch）结束时，系统会销毁所有用于加载数据的工作进程
    并在下一轮开始时重新创建它们。这种频繁的销毁和创建会消耗额外的时间。
    开启 persistent_workers=True 后，这些进程在两轮训练之间会保持存活状态，
    从而消除进程重置的开销，加快整体训练速度，代价是会持续占用部分内存。
    '''
    if config.NUM_WORKERS > 0 and config.PERSISTENT_WORKERS:
        dataloader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs     #解包上面的dataloader_kwargs字典，即载入batch_size,num_workers和pin_memory
    )

    # 验证集
    valid_dataset = CrossLingualDataset(
        valid_data,
        tokenizer,
        config.MAX_LENGTH
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    # 测试集
    test_dataset = CrossLingualDataset(
        test_data,
        tokenizer,
        config.MAX_LENGTH
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    print(f"✓ 训练批次数: {len(train_loader)}")
    print(f"✓ 验证批次数: {len(valid_loader)}")
    print(f"✓ 测试批次数: {len(test_loader)}")
    print(f"✓ DataLoader配置: workers={config.NUM_WORKERS}, "
          f"pin_memory={config.PIN_MEMORY and config.DEVICE.type == 'cuda'}")
    print()

    # ========== 步骤4: 初始化SimCSE-ABSA模型 ==========
    print("=" * 70)
    print("步骤4: 初始化 SimCSE-ABSA 模型")
    print("=" * 70)

    model = XLMSentimentModel(
        model_path=config.MODEL_PATH,
        num_classes=config.NUM_CLASSES,
        projection_dim=config.PROJECTION_DIM,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)

    print_model_info(model)

    # ========== 步骤5: 训练 ==========
    print("=" * 70)
    print("步骤5: 开始训练")
    print("=" * 70)

    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        test_loader,
        config
    )
    trainer.train()

    print("\n" + "=" * 70)
    print("SimCSE-ABSA 项目运行完成！")
    print("=" * 70)
    print(f"\n检查点保存位置: {config.CHECKPOINT_DIR}")
    print(f"日志保存位置: {config.LOG_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()