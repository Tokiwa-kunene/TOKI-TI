"""
数据集模块 - NeighXLM 版本
主要变更：
1. 新增对【语义邻居 (Semantic Neighbor)】文本的支持
   - Anchor: 原始句子（例如：日语训练文本）
   - Neighbor: 离线挖掘的跨语言/同语言语义相似句（例如：英语对齐句）
2. 新增对【逆排名权重 (Inverse Rank-based Weight)】的支持
   - 每个邻居对应一个权重值 w ∈ (0, 1]
   - 权重越低，说明该邻居与 Anchor 的语义相似度越差（噪声邻居）
   - 在对比损失计算中用于动态降低噪声邻居的贡献
3. 向后兼容设计：若 data 字典中未提供 neighbor_text / neighbor_weight，
   自动降级为原 SimCSE 模式（用 Anchor 本身作为正样本，权重全为 1.0）
"""
import torch
from torch.utils.data import Dataset


class CrossLingualDataset(Dataset):
    """跨语言情感分析数据集 - NeighXLM版本"""

    def __init__(self, data, tokenizer, max_length):
        """
        初始化数据集

        Args:
            data (dict): 数据字典，键值说明：
                必需键：
                  'text'            (List[str])   : Anchor 文本列表（原始输入句子）
                  'label'           (List[int])   : 情感标签列表（0=负面, 1=正面 等）
                可选键（NeighXLM 模式）：
                  'neighbor_text'   (List[str])   : 离线挖掘的语义邻居文本列表
                  'neighbor_weight' (List[float]) : 每个邻居的逆排名权重列表，范围 (0, 1]
                  ────────────────────────────────────────────────────────────────
                  逆排名权重计算说明：
                    在离线邻居挖掘阶段（如 FAISS 向量检索），对每个 Anchor 检索 Top-K 邻居
                    第 k 个邻居的逆排名权重定义为：w_k = 1 / k
                    例如 Top-1 邻居权重=1.0，Top-2 权重=0.5，Top-3 权重≈0.33
                    可在 data_preprocessing.py 的 prepare_datasets() 中预先计算并写入字典
                  ────────────────────────────────────────────────────────────────
            tokenizer: XLM-RoBERTa（或其他 AutoTokenizer）实例
            max_length (int): 序列最大长度，超长截断，不足补 padding
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ── 检查是否启用 NeighXLM 模式 ──────────────────────────────────────
        # 若 data 中存在非空的 'neighbor_text' 字段，则进入 NeighXLM 模式
        self.has_neighbor = (
            'neighbor_text' in data
            and data['neighbor_text'] is not None
            and len(data['neighbor_text']) > 0
        )

        # 若 data 中存在非空的 'neighbor_weight' 字段，则启用动态权重
        self.has_weight = (
            'neighbor_weight' in data
            and data['neighbor_weight'] is not None
            and len(data['neighbor_weight']) > 0
        )

        # 打印初始化信息，方便调试
        n = len(self.data['label'])
        mode_str = "NeighXLM模式（跨语言语义邻居）" if self.has_neighbor else "SimCSE降级模式（用原文本作为自身正样本）"
        weight_str = "已启用逆排名权重" if self.has_weight else "未提供权重，全部默认为 1.0"
        print(f"  [CrossLingualDataset] 共 {n} 条样本 | {mode_str} | {weight_str}")

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.data['label'])

    def __getitem__(self, idx):
        """
        获取单个训练样本，返回供 DataLoader 打包的字典

        Args:
            idx (int): 样本索引

        Returns:
            dict 包含以下 key：
              input_ids               : Anchor 的 token id 序列 [max_length]
              attention_mask          : Anchor 的注意力掩码     [max_length]
              neighbor_input_ids      : Neighbor 的 token id 序列 [max_length]
              neighbor_attention_mask : Neighbor 的注意力掩码     [max_length]
              neighbor_weight         : 该邻居对的逆排名权重（标量 float tensor）
              label                   : 情感标签（标量 long tensor）
        """
        # ── (1) 获取 Anchor 文本 ────────────────────────────────────────────
        anchor_text = self.data['text'][idx]        # 原始句子，作为对比学习的锚点
        label = self.data['label'][idx]             # 情感标签

        # ── (2) 获取 Neighbor 文本（语义邻居）──────────────────────────────
        if self.has_neighbor:
            # NeighXLM 模式：使用离线挖掘的跨语言语义相似句作为正样本
            # 例：日语 Anchor → 对应的英语对齐句 Neighbor
            neighbor_text = self.data['neighbor_text'][idx]
        else:
            # SimCSE 降级模式：将 Anchor 本身作为 Neighbor
            # 在 Trainer 的两次 forward 中，dropout 差异构成正样本对
            neighbor_text = anchor_text

        # ── (3) 获取逆排名权重 ──────────────────────────────────────────────
        # 权重越接近 0：该邻居是噪声邻居，应降低其在对比损失中的贡献
        # 权重越接近 1：该邻居语义可靠，正常参与对比学习
        if self.has_weight:
            neighbor_weight = float(self.data['neighbor_weight'][idx])
        else:
            neighbor_weight = 1.0   # 无权重信息时默认全权重（不做过滤）

        # ── (4) Tokenize Anchor 与 Neighbor ─────────────────────────────────
        anchor_encoding = self._tokenize(anchor_text)
        neighbor_encoding = self._tokenize(neighbor_text)

        return {
            # Anchor（锚点）编码结果
            'input_ids': anchor_encoding['input_ids'],
            'attention_mask': anchor_encoding['attention_mask'],

            # Neighbor（语义邻居）编码结果
            # 注意：在 NeighXLM 模式下，这是跨语言的真实语义邻居
            #       在 SimCSE 降级模式下，这与 Anchor 文本相同（依赖 dropout 区分）
            'neighbor_input_ids': neighbor_encoding['input_ids'],
            'neighbor_attention_mask': neighbor_encoding['attention_mask'],

            # 逆排名权重：用于对比损失计算时对噪声邻居进行动态降权
            # 在 _compute_neighxlm_tncse_loss() 中作为每样本的损失权重使用
            'neighbor_weight': torch.tensor(neighbor_weight, dtype=torch.float),

            # 情感分类标签
            'label': torch.tensor(label, dtype=torch.long)
        }

    def _tokenize(self, text):
        """
        对单条文本进行 Tokenize 处理

        词语级别的处理流程：
          原始文本 → 分词 → 添加 [CLS]/[SEP] → 按字典转换为 token id
          不足 max_length 则 padding，超过则 truncation

        Args:
            text (str): 原始输入文本

        Returns:
            dict:
              'input_ids'      : 1D LongTensor [max_length]，token id 序列
              'attention_mask' : 1D LongTensor [max_length]，真实位置为1，padding位置为0
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,     # 超过此长度进行截断
            padding='max_length',           # 不足此长度填充 [PAD] token
            truncation=True,                # 启用截断
            return_tensors='pt'             # 返回 PyTorch tensor 格式
        )

        return {
            # squeeze(0)：去除 batch 维度，将 [1, max_length] 变为 [max_length]
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
