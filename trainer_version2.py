"""
训练器模块 - NeighXLM + TNCSE + I2OA 版本
在 SimCSE-ABSA (trainer_version1.py) 基础上的核心变更：

【修改一：NeighXLM — 数据输入层】
  原 SimCSE：同一 Anchor 两次 forward，利用 Dropout 差异生成正样本对
  新 NeighXLM：Anchor 与离线挖掘的跨语言语义邻居分别 forward，构成跨语言正样本对
  动态权重：每个邻居对配有逆排名权重 w ∈ (0,1]，用于对比损失的加权，过滤噪声邻居

【修改二：TNCSE — 张量范数约束（对比损失层）】
  在 InfoNCE 损失之上叠加 Tensor Norm Constraint 惩罚项：
    L_TN(h, h+) = ||h - h+|| / (||h|| + ||h+||)
  物理意义：在欧氏范数层面拉近 Anchor 与 Neighbor 的表征，与余弦相似度互补
  同样使用逆排名权重对每样本的 L_TN 进行加权，以削弱噪声邻居的影响
  总对比损失：L_CL = L_InfoNCE_weighted + β_tn * L_TN_weighted

【修改三：I2OA — 正交正则化（分类损失层）】
  防止多类别分类头的特征空间坍塌，对最终线性层权重矩阵 W 施加类间正交约束：
    Step 1: 归一化每个类别判别向量 w_i = w_i / ||w_i||
    Step 2: 计算类间内积矩阵 S_ij = w_i^T · w_j
    Step 3: 对严格上三角中的正值求和（仅惩罚锐角，即重叠方向）：
            L_sim = Σ_{i<j} S_ij · I(S_ij > 0)
    Step 4: 加入总分类损失：L_CLS_total = L_CE + λ_i2oa * L_sim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# 将 AdamW 替换为 PyTorch 原生版本
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import os
import time
from datetime import timedelta


class Trainer:
    """SimCSE-ABSA 训练器 - NeighXLM + TNCSE + I2OA 版本"""

    def __init__(self, model, train_loader, valid_loader, test_loader, config):
        """
        初始化训练器

        Args:
            model       : XLMSentimentModel 实例（需实现 get_classifier_weight() 方法）
            train_loader: 训练集 DataLoader（batch 中需含 neighbor_input_ids 等 NeighXLM 字段）
            valid_loader: 验证集 DataLoader
            test_loader : 测试集 DataLoader
            config      : Config 实例，需包含以下属性：
                          LEARNING_RATE, NUM_EPOCHS, WARMUP_RATIO, TEMPERATURE, ALPHA,
                          MAX_GRAD_NORM, DEVICE, NUM_CLASSES, VALID_LANG, TEST_LANG,
                          CHECKPOINT_DIR, SAVE_LAST_EPOCH
                          新增（若未定义则使用默认值）：
                            BETA_TN    (float, default=0.1) : TNCSE 张量范数约束权重
                            LAMBDA_I2OA(float, default=0.01): I2OA 正交正则化权重
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config

        # ── 优化器 ─────────────────────────────────────────────────────────
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE
        )

        # ── 学习率调度器（线性预热 + 线性衰减）────────────────────────────
        total_steps = len(train_loader) * config.NUM_EPOCHS
        warmup_steps = int(config.WARMUP_RATIO * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # ── 损失相关超参数 ──────────────────────────────────────────────────
        self.cls_criterion = nn.CrossEntropyLoss()      # 分类交叉熵损失
        self.temperature = config.TEMPERATURE           # InfoNCE 温度系数
        self.alpha = config.ALPHA                       # 对比损失在总损失中的比重

        # TNCSE：张量范数约束权重 β_tn
        # 控制 L_TN 在对比损失中的贡献比例，建议范围 [0.05, 0.2]
        self.beta_tn = getattr(config, 'BETA_TN', 0.1)

        # I2OA：正交正则化权重 λ_i2oa
        # 控制 L_sim 在分类损失中的贡献比例，建议范围 [0.005, 0.05]
        self.lambda_i2oa = getattr(config, 'LAMBDA_I2OA', 0.01)

        # ── 打印损失权重配置 ────────────────────────────────────────────────
        print(f"\n损失权重配置:")
        print(f"  Alpha (分类 vs 对比)     = {self.alpha}")
        print(f"  ├── 分类损失比重          = {1 - self.alpha:.3f}")
        print(f"  │     └── [I2OA] λ        = {self.lambda_i2oa}")
        print(f"  └── 对比损失比重          = {self.alpha:.3f}")
        print(f"        └── [TNCSE] β_tn    = {self.beta_tn}")

        # ── 历史最佳指标 ────────────────────────────────────────────────────
        self.best_stats = {
            'valid': {'acc': 0.0, 'acc_epoch': 0, 'f1': 0.0, 'f1_epoch': 0},
            'test':  {'acc': 0.0, 'acc_epoch': 0, 'f1': 0.0, 'f1_epoch': 0}
        }
        self.best_valid_f1 = 0.0

        # 时间统计
        self.epoch_times = []
        self.total_start_time = None

    # ═══════════════════════════════════════════════════════════════════════════
    #  核心损失函数
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_neighxlm_tncse_loss(self, anchor_features, neighbor_features, raw_anchor, raw_neighbor,
                                     neighbor_weights, labels): # 注意这里新增了 labels 参数
        device = anchor_features.device
        batch_size = anchor_features.size(0)

        # =====================================================================
        # 1. 计算有监督 InfoNCE (Supervised InfoNCE) - 解决 False Negative 灾难
        # =====================================================================
        # 计算相似度矩阵: [batch_size, batch_size]
        sim_matrix_a2n = torch.matmul(anchor_features, neighbor_features.T) / self.temperature
        sim_matrix_n2a = sim_matrix_a2n.T  # 矩阵转置，计算反向相似度

        # 构建标签掩码：查找 Batch 内情感标签相同的样本对
        # label_mask[i, j] = True 表示样本 i 和 j 属于同一类 (比如都是正面情感)
        label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

        # 对角线是绝对正样本 (Anchor <-> 对应的 Neighbor)，必须保留
        eye = torch.eye(batch_size, dtype=torch.bool, device=device)

        # 找到“假阴性”样本：标签相同，但不是自己对应的 Neighbor
        mask_false_negatives = label_mask & (~eye)

        # 将这些同类样本在相似度矩阵中的得分设为极小值 (-1e9)
        # 这样在后续 logsumexp 计算分母时，它们的 e^(-1e9) 会趋近于 0，不再作为负样本推开
        sim_matrix_a2n.masked_fill_(mask_false_negatives, -1e9)
        sim_matrix_n2a.masked_fill_(mask_false_negatives, -1e9)

        # 手动计算 InfoNCE 损失
        pos_sim_a2n = torch.diag(sim_matrix_a2n) # 提取对角线正样本得分
        pos_sim_n2a = torch.diag(sim_matrix_n2a)

        log_denominator_a2n = torch.logsumexp(sim_matrix_a2n, dim=1)
        log_denominator_n2a = torch.logsumexp(sim_matrix_n2a, dim=1)

        loss_a2n = - (pos_sim_a2n - log_denominator_a2n)
        loss_n2a = - (pos_sim_n2a - log_denominator_n2a)

        # 双向平均
        per_sample_infonce = (loss_a2n + loss_n2a) / 2.0

        # =====================================================================
        # 2. 计算 TNCSE (张量范数约束) - 解决原生 norm 带来的 NaN 梯度爆炸
        # =====================================================================
        # 加入 1e-8 防止距离极小时 torch.norm 求导产生 1/0 的无穷大梯度
        diff_sq = torch.sum((raw_anchor - raw_neighbor) ** 2, dim=1)
        diff_norm = torch.sqrt(diff_sq + 1e-8)

        anchor_norm = torch.sqrt(torch.sum(raw_anchor ** 2, dim=1) + 1e-8)
        neighbor_norm = torch.sqrt(torch.sum(raw_neighbor ** 2, dim=1) + 1e-8)

        per_sample_tn = diff_norm / (anchor_norm + neighbor_norm + 1e-8)

        # =====================================================================
        # 3. 施加逆排名权重并求均值
        # =====================================================================
        loss_infonce_weighted = (per_sample_infonce * neighbor_weights).mean()
        loss_tn_weighted = (per_sample_tn * neighbor_weights).mean()

        total_cl_loss = loss_infonce_weighted + self.config.BETA_TN * loss_tn_weighted

        return total_cl_loss, loss_infonce_weighted, loss_tn_weighted

    def _compute_original_infonce_loss(self, anchor_features, neighbor_features):
        """
        最原始的、标准的 SimCSE (InfoNCE) 损失函数
        (作为 Baseline 供消融实验对比使用)
        """
        device = anchor_features.device
        batch_size = anchor_features.size(0)

        # 生成对角线标签 [0, 1, 2, ..., batch_size-1]
        labels_cl = torch.arange(batch_size, device=device)

        # 计算余弦相似度矩阵
        sim_matrix_a2n = torch.matmul(anchor_features, neighbor_features.T) / self.temperature
        sim_matrix_n2a = sim_matrix_a2n.T

        # 原始的 InfoNCE 就是直接算交叉熵
        loss_a2n = F.cross_entropy(sim_matrix_a2n, labels_cl)
        loss_n2a = F.cross_entropy(sim_matrix_n2a, labels_cl)

        loss_infonce = (loss_a2n + loss_n2a) / 2.0

        # 原始模式下没有 TNCSE 惩罚，因此用 0 占位，保证返回格式统一
        loss_tn = torch.tensor(0.0, device=device)

        return loss_infonce, loss_infonce, loss_tn

    def _compute_i2oa_loss(self, classifier_weight):
        """
        计算 I2OA 正交正则化损失（修改三的核心实现）

        【算法步骤】
          设 W ∈ R^{num_classes × hidden_size} 为分类层权重矩阵
          每行 w_i ∈ R^{hidden_size} 代表第 i 个类别的判别方向向量

          Step 1: 对每行进行 L2 归一化（将每个判别向量投影到单位球面）
                    w_i_norm = w_i / ||w_i||
                  目的：消除模长的影响，只比较向量方向

          Step 2: 计算类间内积矩阵 S ∈ R^{num_classes × num_classes}
                    S_ij = w_i_norm^T · w_j_norm  （即方向余弦）
                  S_ij > 0：两类别向量成锐角（方向部分重叠）→ 需要惩罚
                  S_ij < 0：两类别向量成钝角（方向已分离）→ 不需惩罚
                  S_ij = 0：两类别向量正交（理想状态）

          Step 3: 仅对严格上三角（i < j）中的正值求和（避免重复计数和自相关）
                    L_sim = Σ_{i<j} S_ij · I(S_ij > 0)
                  含义：累加所有"成锐角"的类别对的重叠程度

          Step 4: 返回 L_sim，由调用方乘以 λ_i2oa 后加入总分类损失

        【数学直觉】
          二分类（num_classes=2）时：
            W = [w_0, w_1]（两行）
            若 w_0 和 w_1 方向接近，则模型对两类样本会产生相近的激活值 → 易混淆
            I2OA 惩罚这种重叠，推动 w_0 ⊥ w_1，让两类别的判别方向正交分离

        Args:
            classifier_weight (Tensor): 最终线性分类层权重 W, 形状 [num_classes, hidden_size]

        Returns:
            loss_i2oa (Tensor): 标量，正交正则化损失 L_sim（未乘 λ）
        """
        # Step 1: L2 归一化每个类别向量（沿 dim=1，即 hidden_size 维度）
        # w_norm[i, :] = W[i, :] / ||W[i, :]||
        # 形状：[num_classes, hidden_size]，每行的 L2 范数为 1
        w_norm = F.normalize(classifier_weight, p=2, dim=1)

        # Step 2: 计算类间内积矩阵（归一化后点积 = 余弦相似度）
        # S[i, j] = w_norm[i] · w_norm[j] = cos(w_i, w_j)
        # 形状：[num_classes, num_classes]，对角线元素恒为 1（自身与自身）
        S = torch.matmul(w_norm, w_norm.T)

        # Step 3: 提取严格上三角矩阵（i < j 的元素，避免对角线和重复计数）
        # torch.triu(S, diagonal=1)：保留主对角线以上（不含对角线）的元素，其余置 0
        # 形状：[num_classes, num_classes]，只有上三角非零
        S_upper = torch.triu(S, diagonal=1)

        # 仅保留正值（S_ij > 0 表示两向量成锐角，需要惩罚）
        # 负值（钝角）说明向量已经在反方向，不需额外惩罚，使用 clamp 清零
        # S_positive_upper[i, j] = S_upper[i, j] if S_upper[i, j] > 0 else 0
        S_positive_upper = S_upper.clamp(min=0.0)

        # Step 4: 对所有正值求和得到标量损失
        # 求和而非均值：确保类别数量变化时惩罚尺度合理（配合 λ 调节）
        loss_i2oa = S_positive_upper.sum()

        return loss_i2oa

    # ═══════════════════════════════════════════════════════════════════════════
    #  训练与评估
    # ═══════════════════════════════════════════════════════════════════════════

    def train_epoch(self, epoch):
        """
        训练一个 Epoch（NeighXLM + TNCSE + I2OA 版本）

        主要流程变化（相比 trainer_version1）：
          1. 从 batch 中额外读取 neighbor_input_ids, neighbor_attention_mask, neighbor_weight
          2. Anchor 和 Neighbor 分别经历独立的 forward pass（不再是同文本两次 forward）
          3. 对比损失改用 _compute_neighxlm_tncse_loss() 计算（含加权 InfoNCE + TNCSE）
          4. 分类损失改用 _compute_i2oa_loss() 附加正交正则化项

        Args:
            epoch (int): 当前 epoch 编号（从 0 开始）

        Returns:
            avg_loss       (float): 平均总损失
            avg_cls_loss   (float): 平均分类损失（含 I2OA）
            avg_cl_loss    (float): 平均对比损失（含 TNCSE）
            avg_tn_loss    (float): 平均 TNCSE L_TN 损失（仅供日志监控）
            avg_i2oa_loss  (float): 平均 I2OA L_sim 损失（仅供日志监控）
            epoch_time     (float): 本 epoch 耗时（秒）
        """
        self.model.train()  # 开启训练模式，确保 Dropout 激活

        # 累计损失统计
        total_loss = 0.0
        total_cls_loss = 0.0
        total_cl_loss = 0.0
        total_tn_loss = 0.0    # 仅 TNCSE L_TN 部分，用于日志
        total_i2oa_loss = 0.0  # 仅 I2OA L_sim 部分，用于日志

        epoch_start_time = time.time()

        progress_bar = tqdm(
            self.train_loader,
            desc=f"训练 Epoch {epoch + 1}/{self.config.NUM_EPOCHS}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # ── (1) 将数据搬运至目标设备（GPU/CPU）────────────────────────────
            # Anchor 编码
            anchor_ids = batch['input_ids'].to(self.config.DEVICE)
            anchor_mask = batch['attention_mask'].to(self.config.DEVICE)

            # Neighbor 编码（NeighXLM 新增字段）
            # 在 NeighXLM 模式下：来自跨语言语义邻居
            # 在 SimCSE 降级模式下：与 Anchor 相同的文本（依赖 Dropout 产生差异）
            neighbor_ids = batch['neighbor_input_ids'].to(self.config.DEVICE)
            neighbor_mask = batch['neighbor_attention_mask'].to(self.config.DEVICE)

            # 逆排名权重（NeighXLM 新增字段，SimCSE 降级时全为 1.0）
            neighbor_weights = batch['neighbor_weight'].to(self.config.DEVICE)  # [batch_size]

            labels = batch['label'].to(self.config.DEVICE)

            # ── (2) Anchor Forward Pass ────────────────────────────────────────
            logits_anchor, features_anchor, raw_anchor = self.model(anchor_ids, anchor_mask)

            # ── (3) Neighbor Forward Pass ─────────────────────────────────────
            _, features_neighbor, raw_neighbor = self.model(neighbor_ids, neighbor_mask)

            # ── (4) 分类损失 + I2OA 正交正则化 ───────────────────────────────
            # 标准交叉熵分类损失
            loss_ce = self.cls_criterion(logits_anchor, labels)

            # I2OA：获取最终分类层权重矩阵 W [num_classes, hidden_size]
            # get_classifier_weight() 返回参数引用（带梯度），不是拷贝
            classifier_w = self.model.get_classifier_weight()

            # 计算类间正交正则化损失 L_sim
            loss_i2oa = self._compute_i2oa_loss(classifier_w)

            # 总分类损失 = CE + λ * L_sim
            loss_cls = loss_ce + self.lambda_i2oa * loss_i2oa

            # ── (5) 对比损失 = 双向 InfoNCE + β_tn * 加权 L_TN ──────────────
            if self.config.USE_NEW_LOSS:
                # 启用新损失函数 (带防爆机制和 TNCSE 约束)
                loss_cl, loss_infonce, loss_tn = self._compute_neighxlm_tncse_loss(
                    anchor_features=features_anchor,
                    neighbor_features=features_neighbor,
                    raw_anchor=raw_anchor,
                    raw_neighbor=raw_neighbor
                )
            else:
                # 降级为原始的标准 SimCSE 损失函数
                loss_cl, loss_infonce, loss_tn = self._compute_original_infonce_loss(
                    anchor_features=features_anchor,
                    neighbor_features=features_neighbor
                )
            # ── (6) 总损失（与 version1 相同的加权公式）─────────────────────
            # L_total = (1 - α) * L_cls_total + α * L_cl_total
            loss = (1.0 - self.alpha) * loss_cls + self.alpha * loss_cl

            # ── (7) 反向传播与参数更新 ────────────────────────────────────────
            self.optimizer.zero_grad()      # 清空上一步残留梯度
            loss.backward()                 # 计算当前步所有参数的梯度
            torch.nn.utils.clip_grad_norm_( # 梯度裁剪，防止梯度爆炸
                self.model.parameters(),
                max_norm=self.config.MAX_GRAD_NORM
            )
            self.optimizer.step()           # 更新模型参数
            self.scheduler.step()           # 更新学习率

            # ── (8) 损失累计 ──────────────────────────────────────────────────
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_cl_loss += loss_cl.item()
            total_tn_loss += loss_tn.item()
            total_i2oa_loss += loss_i2oa.item()

            # 更新进度条显示
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{loss_cls.item():.4f}',
                'nce': f'{loss_infonce.item():.4f}',
                'tn': f'{loss_tn.item():.4f}',
                'i2oa': f'{loss_i2oa.item():.4f}'
            })

        epoch_time = time.time() - epoch_start_time
        n_batches = len(self.train_loader)

        return (
            total_loss / n_batches,
            total_cls_loss / n_batches,
            total_cl_loss / n_batches,
            total_tn_loss / n_batches,
            total_i2oa_loss / n_batches,
            epoch_time
        )

    def evaluate(self, data_loader, dataset_name="验证集"):
        """评估模型性能"""
        self.model.eval()   # 关闭 Dropout，进入推理模式
        all_preds = []
        all_labels = []

        with torch.no_grad():  # 不记录计算图，节省显存
            for batch in tqdm(data_loader, desc=f"评估{dataset_name}", leave=False):
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                attention_mask = batch['attention_mask'].to(self.config.DEVICE)
                labels = batch['label'].to(self.config.DEVICE)

                logits, _, _ = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                # 👇 就是补上这两行核心代码 👇
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        # 现在的列表里终于有数据了，可以正常计算了！
        accuracy = accuracy_score(all_labels, all_preds)

        # 根据类别数动态选择 F1 计算方式
        if self.config.NUM_CLASSES > 2:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            f1 = f1_score(all_labels, all_preds, average='binary')

        return accuracy, f1

    def save_checkpoint(self, epoch, valid_acc, valid_f1, test_acc, test_f1,
                        is_best=False, is_last=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'valid_accuracy': valid_acc,
            'valid_f1': valid_f1,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'best_valid_f1': self.best_valid_f1,
            'best_stats': self.best_stats,
            # 保存损失超参数，便于复现
            'alpha': self.alpha,
            'temperature': self.temperature,
            'beta_tn': self.beta_tn,
            'lambda_i2oa': self.lambda_i2oa
        }

        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 最佳模型已保存: {best_path}")

        if is_last and self.config.SAVE_LAST_EPOCH:
            last_path = os.path.join(self.config.CHECKPOINT_DIR, 'last_model.pt')
            torch.save(checkpoint, last_path)
            print(f"  ✓ 最后 Epoch 模型已保存: {last_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  完整训练流程
    # ═══════════════════════════════════════════════════════════════════════════

    def train(self):
        """完整训练流程"""
        valid_lang_name = self.config.get_lang_name(self.config.VALID_LANG)
        test_lang_name = self.config.get_lang_name(self.config.TEST_LANG)

        print("\n" + "=" * 70)
        print("开始 NeighXLM + TNCSE + I2OA 训练")
        print("=" * 70)
        print(f"对比策略: NeighXLM（跨语言语义邻居）+ TNCSE（张量范数约束）")
        print(f"分类策略: 交叉熵损失 + I2OA 类间正交正则化")
        print(f"损失公式: L = (1-α)*[L_CE + λ*L_sim] + α*[L_InfoNCE_w + β*L_TN_w]")
        print(f"  α={self.alpha}, λ_i2oa={self.lambda_i2oa}, β_tn={self.beta_tn}")
        print("=" * 70)

        self.total_start_time = time.time()

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print('=' * 70)

            # ── 训练一个 Epoch ────────────────────────────────────────────────
            avg_loss, avg_cls_loss, avg_cl_loss, avg_tn_loss, avg_i2oa_loss, epoch_time = \
                self.train_epoch(epoch)
            self.epoch_times.append(epoch_time)

            # ── 验证 & 测试评估 ───────────────────────────────────────────────
            print(f"\n在验证集({valid_lang_name})上评估...")
            valid_acc, valid_f1 = self.evaluate(self.valid_loader, f"验证集({valid_lang_name})")

            print(f"在测试集({test_lang_name})上评估...")
            test_acc, test_f1 = self.evaluate(self.test_loader, f"测试集({test_lang_name})")

            # ── 更新历史最佳指标 ──────────────────────────────────────────────
            if valid_acc > self.best_stats['valid']['acc']:
                self.best_stats['valid']['acc'] = valid_acc
                self.best_stats['valid']['acc_epoch'] = epoch + 1

            is_best_valid = False
            if valid_f1 > self.best_stats['valid']['f1']:
                self.best_stats['valid']['f1'] = valid_f1
                self.best_stats['valid']['f1_epoch'] = epoch + 1
                self.best_valid_f1 = valid_f1
                is_best_valid = True

            if test_acc > self.best_stats['test']['acc']:
                self.best_stats['test']['acc'] = test_acc
                self.best_stats['test']['acc_epoch'] = epoch + 1

            if test_f1 > self.best_stats['test']['f1']:
                self.best_stats['test']['f1'] = test_f1
                self.best_stats['test']['f1_epoch'] = epoch + 1

            # ── 打印当前 Epoch 结果 ───────────────────────────────────────────
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1} 结果")
            print('=' * 70)
            print(f"训练损失总计:   {avg_loss:.4f}")
            print(f"  ├─ 分类损失:   {avg_cls_loss:.4f}  (CE + I2OA)")
            print(f"  │    └─ I2OA:  {avg_i2oa_loss:.4f}  (λ={self.lambda_i2oa})")
            print(f"  └─ 对比损失:   {avg_cl_loss:.4f}  (InfoNCE + TNCSE)")
            print(f"       └─ TNCSE: {avg_tn_loss:.4f}  (β={self.beta_tn})")

            print(f"\n验证集({valid_lang_name}):")
            print(f"  准确率: {valid_acc:.4f}")
            print(f"  F1分数: {valid_f1:.4f}")

            print(f"\n测试集({test_lang_name} - Zero-shot):")
            print(f"  准确率: {test_acc:.4f}")
            print(f"  F1分数: {test_f1:.4f}")

            # 特殊标记
            markers = []
            if is_best_valid:
                markers.append("🏆 新的最佳验证F1")
            if test_acc == self.best_stats['test']['acc']:
                markers.append("⭐ 测试集Acc新高")
            if test_f1 == self.best_stats['test']['f1']:
                markers.append("⭐ 测试集F1新高")
            if markers:
                print(f"\n" + " | ".join(markers))

            # ── 保存检查点 ─────────────────────────────────────────────────────
            is_last_epoch = (epoch + 1) == self.config.NUM_EPOCHS
            if is_best_valid or is_last_epoch:
                print()
                self.save_checkpoint(
                    epoch, valid_acc, valid_f1, test_acc, test_f1,
                    is_best=is_best_valid,
                    is_last=is_last_epoch
                )

            print(f"本 Epoch 耗时: {timedelta(seconds=int(epoch_time))}")
            print('=' * 70)

        # ── 训练结束总结报告 ──────────────────────────────────────────────────
        total_time = time.time() - self.total_start_time

        print("\n" + "=" * 70)
        print("NeighXLM + TNCSE + I2OA 训练完成！全过程最佳指标报告")
        print("=" * 70)

        print(f"【验证集 ({valid_lang_name})】")
        print(f"  最高 Accuracy : {self.best_stats['valid']['acc']:.4f}"
              f"  (Epoch {self.best_stats['valid']['acc_epoch']})")
        print(f"  最高 F1 Score : {self.best_stats['valid']['f1']:.4f}"
              f"  (Epoch {self.best_stats['valid']['f1_epoch']})")

        print(f"\n【测试集 ({test_lang_name} - Zero-shot)】")
        print(f"  最高 Accuracy : {self.best_stats['test']['acc']:.4f}"
              f"  (Epoch {self.best_stats['test']['acc_epoch']})")
        print(f"  最高 F1 Score : {self.best_stats['test']['f1']:.4f}"
              f"  (Epoch {self.best_stats['test']['f1_epoch']})")

        print("-" * 70)
        print(f"  总训练时间: {timedelta(seconds=int(total_time))}")
        print(f"  模型保存路径: {self.config.CHECKPOINT_DIR}")
        print("=" * 70 + "\n")
