"""
主要作用：
将数据从csv导出到python中（转化为dataframe（df）格式）
根据需求对数据进行筛选和删除
打包成tensor格式的文件，以便其他部分读取和处理

"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class DataPreprocessor:
    """数据预处理类 - SimCSE版本"""

    def __init__(self, config):
        """
        初始化数据预处理器

        Args:
            config: 配置对象
        """
        self.config = config

    def _read_file(self, file_path, expected_lang):
        """根据文件后缀动态读取，并统一规范字段名称"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 1. 动态读取
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"不支持的文件格式，仅支持 .csv 和 .parquet: {file_path}")

        # 2. 字段对齐（如果是刚才那种 Hugging Face 格式的 parquet 数据）
        # 如果数据含有 'sentence' 和 'label'，但没有 'stars'，我们把它转换成你代码需要的格式
        if 'sentence' in df.columns and 'label' in df.columns and 'stars' not in df.columns:
            # 句子列重命名
            df = df.rename(columns={'sentence': 'review_body'})

            # 逆向映射 stars：把原本的 label(0为负, 1为正) 伪装成 stars (1星和5星)
            # 这样就能完美适配你现有的 convert_label 逻辑
            df['stars'] = df['label'].apply(lambda x: 1 if x == 0 else 5)

            # 补充语言列，使其能通过你后面的语言筛选逻辑
            df['language'] = expected_lang

        return df

    def load_and_filter_data(self):
        """
        加载并筛选数据（从train/valid/test三个文件）

        Returns:
            train_en_df: 训练集英语数据
            valid_en_df: 验证集英语数据
            test_ja_df: 测试集日语数据
        """
        print("=" * 50)
        print("步骤1: 加载数据")
        print("=" * 50)

        # 读取训练集，        注：df为DataFrame格式，类似一个python内部的大型Excel
        print("=" * 50)
        print("步骤1: 加载数据")
        print("=" * 50)

        # ============ 修改读取逻辑 ============
        print(f"\n读取训练集: {self.config.TRAIN_DATA_PATH}")
        train_df = self._read_file(self.config.TRAIN_DATA_PATH, self.config.TRAIN_LANG)
        print(f"  ✓ 原始训练数据: {len(train_df)} 条")

        print(f"\n读取验证集: {self.config.VALID_DATA_PATH}")
        valid_df = self._read_file(self.config.VALID_DATA_PATH, self.config.VALID_LANG)
        print(f"  ✓ 原始验证数据: {len(valid_df)} 条")

        print(f"\n读取测试集: {self.config.TEST_DATA_PATH}")
        test_df = self._read_file(self.config.TEST_DATA_PATH, self.config.TEST_LANG)
        print(f"  ✓ 原始测试数据: {len(test_df)} 条")

        train_df['review_body'] = train_df['review_body'].str.replace(r'[\n\r]+', ' ', regex=True)
        valid_df['review_body'] = valid_df['review_body'].str.replace(r'[\n\r]+', ' ', regex=True)
        test_df['review_body'] = test_df['review_body'].str.replace(r'[\n\r]+', ' ', regex=True)

        # 检查必需的列是否存在
        required_columns = ['stars', 'review_body', 'language']
        for df_name, df in [('训练集', train_df), ('验证集', valid_df), ('测试集', test_df)]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{df_name}缺少必需的列: {missing_cols}")

        if self.config.USE_THREE_CLASSES:
            # 三分类逻辑
            print(">>> 启用三分类模式: 保留3星数据 (Label: 0=负面, 1=中性, 2=正面)")

            #将star转换为数字
            def convert_label(stars):
                if stars in [1, 2]:
                    return 0  # Negative
                elif stars == 3:
                    return 1  # Neutral (新增)
                elif stars in [4, 5]:
                    return 2  # Positive (原为1，现改为2)
                else:
                    return -1
        else:
            # 二分类逻辑 (原有逻辑)
            print(">>> 启用二分类模式: 丢弃3星数据 (Label: 0=负面, 1=正面)")
            def convert_label(stars):
                if stars in [1, 2]:
                    return 0  # Negative
                elif stars in [4, 5]:
                    return 1  # Positive
                else:
                    return -1  # 3星及其他丢弃

        # 获取语言显示名称
        train_lang_name = self.config.get_lang_name(self.config.TRAIN_LANG)
        valid_lang_name = self.config.get_lang_name(self.config.VALID_LANG)
        test_lang_name = self.config.get_lang_name(self.config.TEST_LANG)

        print("\n" + "=" * 50)
        print(f"步骤1.1: 处理训练集 (筛选{train_lang_name})")
        print("=" * 50)
        # === 动态筛选 ===
        train_df = train_df[train_df['language'] == self.config.TRAIN_LANG].copy()
        #[train_df['language'] == self.config.TRAIN_LANG]：判断df中lang列中的值是否与config设置相同，同为True不同为False
        #train_df[上述代码]:保留[]内为True的行，删去False的行
        #.copy()的作用：开辟新的储存空间，储存处理后的df文件，并删掉原先的df文件

        train_df['label'] = train_df['stars'].apply(convert_label)
        train_df = train_df[train_df['label'] != -1].reset_index(drop=True)
        print(f"  筛选{train_lang_name}并转换标签后: {len(train_df)} 条")

        print("\n" + "=" * 50)
        print(f"步骤1.2: 处理验证集 (筛选{valid_lang_name})")
        print("=" * 50)
        # === 动态筛选 ===
        valid_df = valid_df[valid_df['language'] == self.config.VALID_LANG].copy()

        valid_df['label'] = valid_df['stars'].apply(convert_label)
        valid_df = valid_df[valid_df['label'] != -1].reset_index(drop=True)
        print(f"  筛选{valid_lang_name}并转换标签后: {len(valid_df)} 条")

        print("\n" + "=" * 50)
        print(f"步骤1.3: 处理测试集 (筛选{test_lang_name})")
        print("=" * 50)
        # === 动态筛选 ===
        test_df = test_df[test_df['language'] == self.config.TEST_LANG].copy()

        test_df['label'] = test_df['stars'].apply(convert_label)
        test_df = test_df[test_df['label'] != -1].reset_index(drop=True)
        print(f"  筛选{test_lang_name}并转换标签后: {len(test_df)} 条")

        # 在 test_df 语言筛选完成后、USE_SMALL_DATASET 判断之前，加入：

        # ============ 新增：按"列1"文本长度过滤 ============
        if hasattr(self.config, 'MAX_TEXT_LENGTH') and self.config.MAX_TEXT_LENGTH is not None:
            col = '列1'
            print("\n" + "=" * 50)
            print(f"步骤1.4: 按'{col}'过滤 (保留 <= {self.config.MAX_TEXT_LENGTH} 的行)")
            print("=" * 50)
            before = (len(train_df), len(valid_df), len(test_df))
            train_df = train_df[
                pd.to_numeric(train_df[col], errors='coerce').fillna(999) <= self.config.MAX_TEXT_LENGTH].reset_index(
                drop=True)
            valid_df = valid_df[
                pd.to_numeric(valid_df[col], errors='coerce').fillna(999) <= self.config.MAX_TEXT_LENGTH].reset_index(
                drop=True)
            test_df = test_df[
                pd.to_numeric(test_df[col], errors='coerce').fillna(999) <= self.config.MAX_TEXT_LENGTH].reset_index(
                drop=True)
            print(f"  训练集: {before[0]} -> {len(train_df)} 条")
            print(f"  验证集: {before[1]} -> {len(valid_df)} 条")
            print(f"  测试集: {before[2]} -> {len(test_df)} 条")
        # ===================================================

        # 如果使用小规模数据集，进行采样
        if self.config.USE_SMALL_DATASET:
            print("\n" + "=" * 50)
            print("⚠️  启用小规模数据集模式")
            print("=" * 50)

            train_df = self._sample_data(
                train_df,
                self.config.TRAIN_SAMPLE_SIZE,
                "训练集"
            )
            valid_df = self._sample_data(
                valid_df,
                self.config.VALID_SAMPLE_SIZE,
                "验证集"
            )
            test_df = self._sample_data(
                test_df,
                self.config.TEST_SAMPLE_SIZE,
                "测试集"
            )

        print("\n" + "=" * 50)
        print("数据加载完成！")
        print("=" * 50)
        print(f"✓ 最终训练集: {len(train_df)} 条")
        print(f"✓ 最终验证集: {len(valid_df)} 条")
        print(f"✓ 最终测试集: {len(test_df)} 条")
        print("=" * 50 + "\n")

        return train_df, valid_df, test_df

    def _sample_data(self, df, sample_size, dataset_name):
        """
        从数据框中采样指定数量的数据

        Args:
            df: 数据框
            sample_size: 采样数量
            dataset_name: 数据集名称（用于日志）

        Returns:
            采样后的数据框
        """
        if len(df) <= sample_size:
            print(f"\n{dataset_name}:")
            print(f"  数据量({len(df)})小于采样量({sample_size})，使用全部数据")
            label_counts = df['label'].value_counts()
            print(f"  标签分布 - 负面: {label_counts.get(0, 0)}, 正面: {label_counts.get(1, 0)}")
            return df

        if self.config.STRATIFIED_SAMPLING:
            # 分层采样，保持正负样本比例
            try:
                sampled_df, _ = train_test_split(
                    df,
                    train_size=sample_size,
                    stratify=df['label'],
                    random_state=self.config.SEED
                )
                print(f"\n{dataset_name}:")
                print(f"  ✓ 分层采样 {sample_size} 条（保持正负比例）")
            except Exception as e:
                # 如果分层采样失败（比如某类样本太少），使用随机采样
                sampled_df = df.sample(
                    n=sample_size,
                    random_state=self.config.SEED
                ).reset_index(drop=True)
                print(f"\n{dataset_name}:")
                print(f"  ⚠ 分层采样失败，使用随机采样 {sample_size} 条")
                print(f"  失败原因: {e}")
        else:
            # 随机采样
            sampled_df = df.sample(
                n=sample_size,
                random_state=self.config.SEED
            ).reset_index(drop=True)
            print(f"\n{dataset_name}:")
            print(f"  ✓ 随机采样 {sample_size} 条")

        # 打印标签分布
        label_counts = sampled_df['label'].value_counts()
        print(f"  标签分布 - 负面: {label_counts.get(0, 0)}, 正面: {label_counts.get(1, 0)}")

        return sampled_df

    def prepare_datasets(self):
        """
        准备训练集、验证集和测试集（支持 NeighXLM 和 SimCSE 降级模式）
        """
        # 加载和筛选数据
        train_df, valid_df, test_df = self.load_and_filter_data()

        print("\n" + "=" * 50)
        print("步骤2: 准备数据集 (检查模式)")
        print("=" * 50)

        # 1. 提取基础数据
        anchor_texts = train_df['review_body'].tolist()
        labels = train_df['label'].tolist()

        # 初始化基础字典
        train_data = {
            'text': anchor_texts,
            'label': labels
        }

        # 2. 动态检查是否含有邻居列，实现 NeighXLM 与 SimCSE 的无缝切换
        if 'neighbor_text' in train_df.columns:
            print("✓ 检测到 'neighbor_text' 列，启用 NeighXLM 跨语言邻居模式")
            neighbor_1_texts = train_df['neighbor_text'].tolist()

            # 检查权重列
            if 'neighbor_weight' in train_df.columns:
                neighbor_1_weights = train_df['neighbor_weight'].tolist()
            else:
                neighbor_1_weights = [1.0] * len(anchor_texts)

            # 检查是否存在第二组邻居 (双邻居扩增)
            if 'neighbor_text_2' in train_df.columns:
                print("✓ 检测到 'neighbor_text_2' 列，启用双邻居数据扩增")
                neighbor_2_texts = train_df['neighbor_text_2'].tolist()

                if 'neighbor_weight_2' in train_df.columns:
                    neighbor_2_weights = train_df['neighbor_weight_2'].tolist()
                else:
                    neighbor_2_weights = [1.0] * len(anchor_texts)

                # 数据翻倍：Anchor 对应两个不同的邻居
                train_data['text'] = anchor_texts + anchor_texts
                train_data['label'] = labels + labels
                train_data['neighbor_text'] = neighbor_1_texts + neighbor_2_texts
                train_data['neighbor_weight'] = neighbor_1_weights + neighbor_2_weights
            else:
                # 只有单组邻居
                train_data['neighbor_text'] = neighbor_1_texts
                train_data['neighbor_weight'] = neighbor_1_weights
        else:
            print("⚠ 未检测到 'neighbor_text' 列，自动降级为标准 SimCSE 模式")
            print("  -> 将使用 Dropout 生成正样本对")
            # 字典中不添加 neighbor 字段，dataset_version2.py 会自动处理这种降级情况

        print("=" * 50 + "\n")

        # 构建验证数据
        valid_data = {
            'text': valid_df['review_body'].tolist(),
            'label': valid_df['label'].tolist()
        }

        # 构建测试数据（日语数据 - Zero-shot测试）
        test_data = {
            'text': test_df['review_body'].tolist(),
            'label': test_df['label'].tolist()
        }

        print("=" * 50)
        print("数据准备完成！")
        print("=" * 50)
        print(f"训练集大小: {len(train_data['label'])} 条")
        print(f"验证集大小: {len(valid_data['label'])} 条")
        print(f"测试集大小: {len(test_data['label'])} 条")
        print("=" * 50 + "\n")

        return train_data, valid_data, test_data