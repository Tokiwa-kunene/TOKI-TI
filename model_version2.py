"""
模型架构定义 - NeighXLM + I2OA 版本
在 SimCSE-ABSA (model_version1.py) 基础上的变更：
1. 暴露分类头最终线性层的权重矩阵 W，供 Trainer 计算 I2OA 正交正则化损失
2. 新增 get_classifier_weight() 方法，返回 W 的引用（非拷贝）
   以确保梯度能正确回传至分类层
3. Backbone、Attention Pooling、投影头等结构保持不变
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AttentionPooling(nn.Module):
    """
    注意力池化层
    使用可学习的注意力机制将序列 [batch, seq_len, hidden] 聚合为句向量 [batch, hidden]
    相比 CLS 直接取用，能更灵活地捕捉情感关键词的位置信息
    """

    def __init__(self, hidden_size):
        """
        Args:
            hidden_size (int): Backbone 的隐藏层维度（XLM-RoBERTa 为 768）
        """
        super(AttentionPooling, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 特征变换：hidden → hidden
            nn.Tanh(),                            # 非线性激活，将注意力分数压缩到 (-1, 1)
            nn.Linear(hidden_size, 1)             # 压缩到标量注意力得分：hidden → 1
        )

    def forward(self, sequence_output, attention_mask=None):
        """
        Args:
            sequence_output : Backbone 全序列输出 [batch_size, seq_len, hidden_size]
            attention_mask  : 填充掩码 [batch_size, seq_len]，padding 位置为 0

        Returns:
            pooled_output     : 句向量 [batch_size, hidden_size]
            attention_weights : 各位置注意力权重 [batch_size, seq_len, 1]
        """
        # 计算每个 token 的原始注意力得分：[batch_size, seq_len, 1]
        attention_scores = self.attention(sequence_output)

        if attention_mask is not None:
            # 在最后增加维度以对齐 attention_scores 的形状：[batch, seq_len, 1]
            attention_mask_expanded = attention_mask.unsqueeze(-1)
            # 对 padding 位置（mask==0）填充极小值，使 softmax 后权重趋近于 0
            attention_scores = attention_scores.masked_fill(
                attention_mask_expanded == 0,
                -1e9
            )

        # 对序列维度做 softmax，得到归一化注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]

        # 加权求和得到最终句向量：sum(token_vec * weight) → [batch, hidden]
        pooled_output = torch.sum(sequence_output * attention_weights, dim=1)

        return pooled_output, attention_weights


class SpecificFeatureExtractor(nn.Module):
    """
    情感分类分支（Branch A）
    包含：Attention Pooling → MLP 分类头

    I2OA 改动：
      新增 get_final_layer_weight() 方法，暴露最终线性层权重矩阵 W
      W 的形状为 [num_classes, hidden_size]，每行 w_i 是第 i 个类别的判别向量
      Trainer 会利用 W 计算类间正交正则化损失（I2OA），防止类别特征空间坍塌
    """

    def __init__(self, hidden_size, num_classes, dropout_rate=0.1):
        """
        Args:
            hidden_size  (int): Backbone 隐藏维度
            num_classes  (int): 分类类别数（2=二分类，3=三分类）
            dropout_rate (float): Dropout 概率
        """
        super(SpecificFeatureExtractor, self).__init__()

        # 注意力池化层：将全序列压缩为句子级表征
        self.attn_pooling = AttentionPooling(hidden_size)

        # MLP 分类头：池化输出 → 最终类别 logits
        # 注意：不要将最终线性层拆出为独立属性，否则 Sequential 索引会变化
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # [0] 特征变换层
            nn.Tanh(),                            # [1] 激活函数
            nn.Dropout(p=dropout_rate),           # [2] 防过拟合
            nn.Linear(hidden_size, num_classes)   # [3] 最终分类层 ← I2OA 目标
        )

    def get_final_layer_weight(self):
        """
        [I2OA 接口] 返回最终线性分类层的权重矩阵 W

        W 的物理含义：
          W[i, :] = 第 i 个类别的判别方向向量（即该类别在特征空间中的"模板"）
          若两行之间的内积 S_ij = W_i · W_j > 0，说明两个类别在特征空间中存在重叠
          I2OA 损失会惩罚这种重叠，推动各类别方向趋向正交，减少误分类

        Returns:
            weight (Tensor): 形状 [num_classes, hidden_size]，带梯度，不是拷贝
        """
        # self.classifier[3] 即 nn.Sequential 中索引为 3 的 nn.Linear 层
        # .weight 直接返回参数引用（非 clone），梯度可正常回传
        return self.classifier[3].weight

    def forward(self, sequence_output, attention_mask):
        """
        Args:
            sequence_output : Backbone 全序列输出 [batch, seq_len, hidden]
            attention_mask  : 填充掩码 [batch, seq_len]

        Returns:
            logits           : 分类 logits [batch, num_classes]
            pooled_output    : 池化后的句向量 [batch, hidden]（供 Branch B 或 I2OA 使用）
            attention_weights: 注意力权重 [batch, seq_len, 1]
        """
        # 注意力池化：将序列压缩为句向量
        pooled_output, attention_weights = self.attn_pooling(sequence_output, attention_mask)

        # MLP 分类头：句向量 → 类别 logits
        logits = self.classifier(pooled_output)

        return logits, pooled_output, attention_weights


class XLMSentimentModel(nn.Module):
    """
    SimCSE-ABSA 模型主体 - NeighXLM + I2OA 版本

    架构（与 model_version1 保持一致，仅暴露 I2OA 接口）：
      ┌─ XLM-RoBERTa (Backbone, Dropout=dropout_rate)
      ├─ Branch A: 情感分类
      │    └── Attention Pooling → MLP Classifier (Cross-Entropy Loss)
      │         ↑ I2OA: 对最终线性层权重 W 施加类间正交正则化
      └─ Branch B: 对比学习
           └── [CLS] → Projection Head → L2 Norm
                ↑ NeighXLM: 以跨语言语义邻居作为正样本对
                ↑ TNCSE:    在 InfoNCE 之上叠加张量范数约束惩罚
    """

    def __init__(self, model_path, num_classes=2, projection_dim=128, dropout_rate=0.1):
        """
        Args:
            model_path    (str)  : Backbone 模型本地路径（XLM-RoBERTa 或 mBERT）
            num_classes   (int)  : 分类类别数
            projection_dim(int)  : 对比学习投影维度
            dropout_rate  (float): Dropout 概率，影响 SimCSE 正样本生成
        """
        super(XLMSentimentModel, self).__init__()

        print("\n" + "=" * 70)
        print("初始化 SimCSE-ABSA 模型 (NeighXLM + I2OA 版本)")
        print("=" * 70)

        # ── Backbone：XLM-RoBERTa / mBERT ────────────────────────────────────
        print(f"\n[1/3] 加载 Backbone (AutoModel)")
        print(f"      模型路径: {model_path}")
        self.backbone = AutoModel.from_pretrained(
            model_path,
            hidden_dropout_prob=dropout_rate,           # 隐藏层 Dropout（SimCSE 正样本核心）
            attention_probs_dropout_prob=dropout_rate   # 注意力矩阵 Dropout
        )
        self.hidden_size = self.backbone.config.hidden_size
        print(f"      ✓ Backbone 加载成功 (hidden_size={self.hidden_size})")
        print(f"      ✓ Dropout Rate: {self.backbone.config.hidden_dropout_prob}")

        # ── Branch A：情感分类分支（含 I2OA 权重暴露接口）────────────────────
        print(f"\n[2/3] 初始化 Branch A: 情感分类分支 (含 I2OA 接口)")
        self.specific_feature_extractor = SpecificFeatureExtractor(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        print(f"      ✓ Attention Pooling + MLP 分类头")
        print(f"      ✓ 分类器结构: {self.hidden_size} → {self.hidden_size} → {num_classes}")
        print(f"      ✓ [I2OA] 最终线性层权重 W: [{num_classes}, {self.hidden_size}]")

        # ── Branch B：对比学习投影头（NeighXLM + TNCSE 在 Trainer 中实现）────
        print(f"\n[3/3] 初始化 Branch B: 对比学习投影头")
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),  # 维度保持变换
            nn.ReLU(),                                      # 非线性激活
            nn.Linear(self.hidden_size, projection_dim)     # 压缩到对比空间
        )
        print(f"      ✓ 投影头结构: {self.hidden_size} → {projection_dim}")
        print(f"      ✓ [NeighXLM] 正样本：跨语言语义邻居（在 Trainer 中使用）")
        print(f"      ✓ [TNCSE]    附加张量范数约束惩罚（在 Trainer 中计算）")

        print("\n" + "=" * 70)
        print("模型初始化完成")
        print("=" * 70 + "\n")

    def get_classifier_weight(self):
        """
        [I2OA 对外接口] 获取最终分类层的权重矩阵 W

        在 Trainer 的 _compute_i2oa_loss() 中调用此方法获取 W，然后：
          1. 对 W 的每行进行 L2 归一化 → w_i = w_i / ||w_i||
          2. 计算类间内积矩阵 S = W_norm @ W_norm.T
          3. 对 S 的严格上三角中正值求和得到 L_sim
          4. 将 λ * L_sim 加入总分类损失

        Returns:
            weight (Tensor): 形状 [num_classes, hidden_size]，带梯度
        """
        return self.specific_feature_extractor.get_final_layer_weight()

    def forward(self, input_ids, attention_mask, return_attention=False):
        """
        统一前向传播（Anchor 或 Neighbor 均通过此函数）

        Args:
            input_ids       : [batch_size, seq_len] 输入 token id
            attention_mask  : [batch_size, seq_len] 注意力掩码
            return_attention : 是否额外返回注意力权重（默认 False）

        Returns:
            logits           : [batch_size, num_classes] 分类 logits (Branch A)
            features         : [batch_size, projection_dim] 对比特征 (Branch B, L2 归一化)
            attention_weights: （可选）[batch_size, seq_len, 1]
        """
        # ── Backbone 编码 ─────────────────────────────────────────────────────
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 全序列表征：[batch, seq_len, hidden]
        sequence_output = outputs.last_hidden_state
        # [CLS] 位置表征（经过 pooler 线性变换）：[batch, hidden]
        cls_output = outputs.pooler_output

        # ── Branch A：情感分类 ────────────────────────────────────────────────
        logits, pooled_features, attention_weights = self.specific_feature_extractor(
            sequence_output, attention_mask
        )

        # ── Branch B：对比学习 ────────────────────────────────────────────────
        # 1. 提取原始特征（用于 TNCSE 计算真实欧氏距离和模长约束）
        raw_features = self.projection_head(cls_output)  # [batch, projection_dim]

        # 2. L2归一化特征（用于 SupCon 计算余弦相似度）
        normalized_features = F.normalize(raw_features, dim=1)

        # 多返回一个 raw_features
        if return_attention:
            return logits, normalized_features, raw_features, attention_weights
        return logits, normalized_features, raw_features
