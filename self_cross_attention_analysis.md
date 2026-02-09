# MARDM Self-Attention 和 Cross-Attention 执行流程详解

## 1. 整体架构

### 1.1 Transformer Block 结构

```
MARTransBlock (每个Transformer层)
├── Self-Attention (Motion tokens之间的注意力)
├── Cross-Attention (Motion tokens → Audio features)
└── MLP (前馈网络)
```

### 1.2 执行顺序

```
输入: x [T_motion, B, C]
  ↓
1. Self-Attention
  ├─ Norm1 (LayerNorm)
  ├─ AdaLN调制 (shift_msa, scale_msa)
  ├─ Self-Attention计算
  └─ 残差连接 (gate_msa)
  ↓
2. Cross-Attention (如果启用)
  ├─ Norm_cross (LayerNorm)
  ├─ AdaLN调制 (shift_msa, scale_msa)
  ├─ Cross-Attention计算 (Motion → Audio)
  └─ 残差连接 (gate_msa)
  ↓
3. MLP
  ├─ Norm2 (LayerNorm)
  ├─ AdaLN调制 (shift_mlp, scale_mlp)
  ├─ MLP前馈
  └─ 残差连接 (gate_mlp)
  ↓
输出: x [T_motion, B, C]
```

## 2. Self-Attention 详细流程

### 2.1 输入准备

```python
# 输入: x [T_motion=300, B, C]
# padding_mask: [B, T_motion=300]

# 检测格式并转换
if x是sequence-first格式:
    x_batch = x.permute(1, 0, 2)  # [300, B, C] -> [B, 300, C]
else:
    x_batch = x  # 已经是 [B, 300, C]
```

### 2.2 Self-Attention 计算

```python
# 1. Layer Normalization
x_norm = self.norm1(x_batch)  # [B, 300, C]

# 2. AdaLN调制 (Adaptive Layer Normalization)
# shift_msa, scale_msa 来自条件向量c
x_modulated = modulate_here(x_norm, shift_msa, scale_msa)  # [B, 300, C]

# 3. Self-Attention
attn_out = self.attn(x_modulated, mask=padding_mask)  # [B, 300, C]
```

### 2.3 Self-Attention 内部实现

```python
# 在Attention类中:
# 输入: x_modulated [B, 300, C]

# 生成Q, K, V (都来自motion tokens)
q = self.query(x_modulated)  # [B, 300, C] -> [B, n_head, 300, head_dim]
k = self.key(x_modulated)    # [B, 300, C] -> [B, n_head, 300, head_dim]
v = self.value(x_modulated)  # [B, 300, C] -> [B, n_head, 300, head_dim]

# Attention矩阵: [B, n_head, 300, 300]
att = (q @ k.transpose(-2, -1)) * (1.0 / sqrt(head_dim))

# 应用padding mask
if padding_mask is not None:
    att = att.masked_fill(padding_mask != 0, float('-inf'))

# Softmax归一化
att = softmax(att, dim=-1)  # 在最后一个维度(300)上归一化

# 加权求和
y = att @ v  # [B, n_head, 300, head_dim] -> reshape -> [B, 300, C]
```

**关键点**:
- Q, K, V 都来自 motion tokens
- Attention矩阵是 `[300, 300]`，每个motion token关注其他motion tokens
- 用于学习motion tokens之间的时序依赖关系

### 2.4 残差连接

```python
# 转换回原始格式
if x是sequence-first:
    attn_out_seq = attn_out.permute(1, 0, 2)  # [B, 300, C] -> [300, B, C]
    gate_for_attn = gate_msa.unsqueeze(0)  # [B, C] -> [1, B, C]
else:
    attn_out_seq = attn_out  # [B, 300, C]
    gate_for_attn = gate_msa.unsqueeze(1)  # [B, C] -> [B, 1, C]

# 残差连接 (带gate控制)
x = x + gate_for_attn * attn_out_seq
```

## 3. Cross-Attention 详细流程

### 3.1 输入准备

```python
# 输入: x [T_motion=300, B, C] (经过self-attention后)
# audio_seq: [T_audio=1000, B, C] (sequence-first格式)

# 转换格式
if x是sequence-first:
    x_batch_first = x.permute(1, 0, 2)  # [300, B, C] -> [B, 300, C]
else:
    x_batch_first = x  # [B, 300, C]

audio_seq_batch_first = audio_seq.permute(1, 0, 2)  # [1000, B, C] -> [B, 1000, C]
```

### 3.2 Cross-Attention 计算

```python
# 1. Layer Normalization
x_norm_cross = self.norm_cross(x_batch_first)  # [B, 300, C]

# 2. AdaLN调制 (使用与self-attention相同的shift_msa, scale_msa)
x_modulated_cross = modulate_here(x_norm_cross, shift_msa, scale_msa)  # [B, 300, C]

# 3. Cross-Attention
x_cross = self.cross_attn(
    x_modulated_cross,      # [B, 300, C] - Motion tokens (Query)
    audio_seq_batch_first,  # [B, 1000, C] - Audio features (Key & Value)
    audio_mask              # [B, 1000] - Audio padding mask
)  # 输出: [B, 300, C]
```

### 3.3 Cross-Attention 内部实现

```python
# 在CrossAttention类中:
# Query: 来自motion tokens
q = self.query(x_modulated_cross)  # [B, 300, C] -> [B, n_head, 300, head_dim]

# Key & Value: 来自audio features
k = self.key(audio_seq_batch_first)    # [B, 1000, C] -> [B, n_head, 1000, head_dim]
v = self.value(audio_seq_batch_first)  # [B, 1000, C] -> [B, n_head, 1000, head_dim]

# Attention矩阵: [B, n_head, 300, 1000]
att = (q @ k.transpose(-2, -1)) * (1.0 / sqrt(head_dim))

# 应用audio mask
if audio_mask is not None:
    att = att.masked_fill(audio_mask != 0, float('-inf'))

# Softmax归一化 (在audio维度上)
att = softmax(att, dim=-1)  # 在最后一个维度(1000)上归一化

# 加权求和
y = att @ v  # [B, n_head, 300, head_dim] -> reshape -> [B, 300, C]
```

**关键点**:
- Query来自motion tokens (300个)
- Key & Value来自audio features (1000个)
- Attention矩阵是 `[300, 1000]`，每个motion token关注所有audio frames
- 用于将音频信息融合到motion tokens中

### 3.4 残差连接

```python
# 转换回原始格式
if x是sequence-first:
    x_cross_seq = x_cross.permute(1, 0, 2)  # [B, 300, C] -> [300, B, C]
    gate_for_cross = gate_msa.unsqueeze(0)  # [B, C] -> [1, B, C]
else:
    x_cross_seq = x_cross  # [B, 300, C]
    gate_for_cross = gate_msa.unsqueeze(1)  # [B, C] -> [B, 1, C]

# 残差连接 (带gate控制)
x = x + gate_for_cross * x_cross_seq
```

## 4. MLP 详细流程

### 4.1 MLP 计算

```python
# 1. Layer Normalization
x_norm_mlp = self.norm2(x)  # [B, 300, C] 或 [300, B, C]

# 2. AdaLN调制 (使用shift_mlp, scale_mlp)
x_modulated_mlp = modulate_here(x_norm_mlp, shift_mlp, scale_mlp)

# 3. MLP前馈
mlp_out = self.mlp(x_modulated_mlp)  # [B, 300, C] 或 [300, B, C]

# 4. 残差连接 (带gate_mlp控制)
x = x + gate_mlp * mlp_out
```

## 5. AdaLN (Adaptive Layer Normalization) 机制

### 5.1 AdaLN参数生成

```python
# 从条件向量c生成6个参数
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
    self.adaLN_modulation(c).chunk(6, dim=1)
# 每个参数: [B, C]
```

### 5.2 AdaLN调制公式

```python
def modulate_here(x, shift, scale):
    """
    Adaptive Layer Normalization调制
    x: [B, T, C] 或 [T, B, C]
    shift: [B, C] 或 [1, B, C]
    scale: [B, C] 或 [1, B, C]
    """
    return x * (1 + scale) + shift
```

**作用**:
- `shift` 和 `scale` 根据条件向量(音频全局特征)动态调整
- 让模型能够根据音频内容调整motion生成

## 6. 完整数据流示例

### 6.1 输入

```python
# Motion latent tokens
latents: [B=16, L=300, ae_dim=512]

# Audio features
audio_seq: [B=16, T_audio=1000, audio_dim=512]

# 条件向量 (音频全局特征的平均)
cond_vector: [B=16, latent_dim=1024]
```

### 6.2 在MARDM.forward()中

```python
# 1. 处理motion latents
x = self.input_process(latents)  # [B, 300, ae_dim] -> [B, 300, latent_dim]
x = self.position_enc(x)         # 添加位置编码
x = x.permute(1, 0, 2)            # [B, 300, C] -> [300, B, C] (sequence-first)

# 2. 处理audio features
audio_seq_processed = self.audio_seq_emb(audio_seq)  # [B, 1000, 512] -> [B, 1000, 1024]
audio_seq_processed = audio_seq_processed.permute(1, 0, 2)  # [1000, B, 1024]

# 3. 通过多个Transformer层
for block in self.MARTransformer:
    x = block(x, cond_vector, padding_mask, audio_seq=audio_seq_processed, audio_mask=None)
    # x: [300, B, C] -> [300, B, C]
```

### 6.3 在MARTransBlock中 (单个层)

```python
# 输入
x: [300, B=16, C=1024]  # sequence-first格式
cond_vector: [B=16, C=1024]
audio_seq: [1000, B=16, C=1024]  # sequence-first格式

# === Self-Attention ===
# 1. 格式转换
x_batch = x.permute(1, 0, 2)  # [300, 16, 1024] -> [16, 300, 1024]

# 2. Norm + AdaLN + Self-Attention
x_norm = self.norm1(x_batch)  # [16, 300, 1024]
x_modulated = modulate_here(x_norm, shift_msa, scale_msa)  # [16, 300, 1024]
attn_out = self.attn(x_modulated, mask=padding_mask)  # [16, 300, 1024]
# Self-Attention内部: Q, K, V都来自x_modulated，计算[300, 300]的attention矩阵

# 3. 残差连接
attn_out_seq = attn_out.permute(1, 0, 2)  # [16, 300, 1024] -> [300, 16, 1024]
gate_for_attn = gate_msa.unsqueeze(0)  # [16, 1024] -> [1, 16, 1024]
x = x + gate_for_attn * attn_out_seq  # [300, 16, 1024]

# === Cross-Attention ===
# 1. 格式转换
x_batch_first = x.permute(1, 0, 2)  # [300, 16, 1024] -> [16, 300, 1024]
audio_seq_batch_first = audio_seq.permute(1, 0, 2)  # [1000, 16, 1024] -> [16, 1000, 1024]

# 2. Norm + AdaLN + Cross-Attention
x_norm_cross = self.norm_cross(x_batch_first)  # [16, 300, 1024]
x_modulated_cross = modulate_here(x_norm_cross, shift_msa, scale_msa)  # [16, 300, 1024]
x_cross = self.cross_attn(
    x_modulated_cross,      # [16, 300, 1024] - Query
    audio_seq_batch_first,  # [16, 1000, 1024] - Key & Value
    audio_mask
)  # [16, 300, 1024]
# Cross-Attention内部: Q来自x_modulated_cross，K&V来自audio_seq_batch_first，计算[300, 1000]的attention矩阵

# 3. 残差连接
x_cross_seq = x_cross.permute(1, 0, 2)  # [16, 300, 1024] -> [300, 16, 1024]
gate_for_cross = gate_msa.unsqueeze(0)  # [16, 1024] -> [1, 16, 1024]
x = x + gate_for_cross * x_cross_seq  # [300, 16, 1024]

# === MLP ===
# 1. 格式转换
x_batch_mlp = x.permute(1, 0, 2)  # [300, 16, 1024] -> [16, 300, 1024]

# 2. Norm + AdaLN + MLP
x_norm_mlp = self.norm2(x_batch_mlp)  # [16, 300, 1024]
x_modulated_mlp = modulate_here(x_norm_mlp, shift_mlp, scale_mlp)  # [16, 300, 1024]
mlp_out = self.mlp(x_modulated_mlp)  # [16, 300, 1024]

# 3. 残差连接
mlp_out_seq = mlp_out.permute(1, 0, 2)  # [16, 300, 1024] -> [300, 16, 1024]
gate_for_mlp = gate_mlp.unsqueeze(0)  # [16, 1024] -> [1, 16, 1024]
x = x + gate_for_mlp * mlp_out_seq  # [300, 16, 1024]

# 输出
x: [300, 16, 1024]  # sequence-first格式
```

## 7. 关键设计特点

### 7.1 Self-Attention的作用
- **学习motion tokens之间的时序依赖**
- 每个motion token关注其他motion tokens
- Attention矩阵: `[300, 300]`
- 用于建模动作的连续性和时序关系

### 7.2 Cross-Attention的作用
- **将音频信息融合到motion tokens**
- Motion tokens作为Query，Audio features作为Key & Value
- Attention矩阵: `[300, 1000]`
- 用于实现音频驱动的动作生成

### 7.3 执行顺序的重要性
1. **先Self-Attention**: 让motion tokens先建立内部关系
2. **再Cross-Attention**: 在已有关系基础上融合音频信息
3. **最后MLP**: 进一步处理和变换特征

### 7.4 AdaLN的作用
- **条件控制**: 根据音频全局特征动态调整每一层的参数
- **6个参数**: 
  - `shift_msa, scale_msa`: 控制self-attention
  - `gate_msa`: 控制self-attention和cross-attention的残差连接
  - `shift_mlp, scale_mlp`: 控制MLP
  - `gate_mlp`: 控制MLP的残差连接

## 8. 总结

MARDM的attention机制采用**串行设计**:
1. **Self-Attention**: Motion tokens内部交互 (`[300, 300]`)
2. **Cross-Attention**: Motion tokens关注Audio features (`[300, 1000]`)
3. **MLP**: 特征变换

这种设计让模型能够:
- 先学习motion的时序结构 (Self-Attention)
- 再将音频信息融合进来 (Cross-Attention)
- 最终生成与音频同步的动作
