# 解码探索队 · OmniEEG 复赛训练与复现

本目录使用同一份公开 CBraMod 初始权重，对主赛道各库、各官方种子分别训练。每折输出最佳候选权重及验证集选模记录。训练以正式开放数据和对应划分为准，公开 rc8 模拟数据用于接口检查。

## 数据与环境

TI-ONE 镜像为 `omni-cr.tencentcloudcr.com/game-image/competition-env:0.0.6`，训练使用镜像内的 PyTorch、NumPy、PyArrow 和 PyLance。[公开评测包环境说明](../public_evaluation/environment/image.txt)

CBraMod 源码固定到作者的 `b9e961003214326972c567eff390e75b0287e32a` 提交，附 MIT 许可。骨干只调整了 Python 导入路径。初始权重来自作者发布，训练配置记录其 SHA-256。云端任务启动前将权重作为普通文件放到可读路径。推理使用导出的骨干和头权重。[CBraMod 源码](https://github.com/wjq-learning/CBraMod/tree/b9e961003214326972c567eff390e75b0287e32a) · [作者权重](https://huggingface.co/weighting666/CBraMod/blob/6d032ec/pretrained_weights.pth)

工作区的 `training/config.example.json` 展示一个任务和一个占位种子。按正式 release metadata 填写十四库的 `lance`、`lance_version`、`subject_column`、通道数、原始窗点数、采样率及有序 `label_values`。所有相对路径以配置文件所在目录为起点。每个官方种子对应两份一维整数 `.npy` 文件，存储该 Lance 版本中零起点的 train 和 validation 行索引。根据官方索引或被试划分生成这两份文件，并记录其来源。如果官方给的是样本 ID，先按 ID 对到对应版本的行位置。训练按这两个集合读数据，并核对被试隔离。按官方通道顺序输入，通道名不参与空间索引。

## 尺度规则和候选

- `physical_uv` 用于能够追溯微伏单位的数据，`uv_per_value` 填数值到微伏的换算系数。 `fixed_multiplier` 用于有依据的同库固定未知倍率。 `unknown` 表示目前不符合旁路资格。放在 `unknown` 的库仍训练 R、N、N_F 和 N_P。只有前两类且本折训练集有至少两个非零窗、对数尺度方差非零时才训练 E 与各 L 路径。
- `reliability_02` 的 1000 Hz 波形按窗逐导以 101 点 Hann 窗低通 FIR（名义截止 80 Hz）抗混叠，随后 5 倍降采样。低通和降采样先于 RMS 分解，尺度解释对应降采样后的有效波形。其余 13 库保持 200 Hz。全部路径共用每秒 200 点的 CBraMod 分块。
- 共同标记依次为有效全零、全缺失、部分缺失、存在非有限值和达到饱和阈值。非有限点填零且从尺度统计中排除，降采样任务的每组五点全部有效才保留下游对应点。只有有依据时才填写交付数值单位下的 `saturation_abs`，未提供时该标记固定为 0。推理接口只提供波形和通道名，交付数据中已经填为有限零且没有额外掩码的采样点按有效点处理。
- R 保留窗间尺度，N 不设显式尺度旁路。E 在输入及位置嵌入之后、首个时空模块之前融合。L 在读出之后、最终 ELU 之前融合。展平头每块压缩至 32 维后加权展平，经 256 维隐层到 200 维。四个冻结候选共享批内骨干前向，展平与池化头各自训练。此实现在线计算池化特征，以利用同批已经计算的 token，暂不写磁盘缓存。
- 各候选从同一份权重出发，按配对样本顺序运行。有效批量 32，通过微批量累积梯度。使用训练折逆频率加权交叉熵和 AdamW，骨干与头的峰值学习率分别为 `1e-5`、`1e-3`，前 10% 更新预热后余弦衰减。CUDA 训练使用 BF16 autocast，验证和推理使用 FP32。

## 云端短测和正式训练

先对形状较大的任务用训练折短测，记录每次更新耗时和显存峰值。根据吞吐选定共同的 50、20 或 10 轮上限。短测只运行训练更新，不读验证折或保存权重。

```bash
python semifinal/training/train.py --config train_config.json --task "$TASK" --seed "$SEED" \
  --backbone /path/to/pretrained_weights.pth --output runs --profile-updates 20
```

正式运行去掉 `--profile-updates 20`。每个任务与种子独立启动，可以将不同组合分配到两张卡。输出目录为 `runs/<任务>/<种子>/`。同一运行目录只使用一次，需要重跑时指定新的输出根目录。

每个候选保留验证平衡准确率最高的权重，同分保留最早轮次。`selection.json` 记录本折的尺度统计、候选得分、正式选择 C 和研究选择 S。C 按 N_P、N_F、N、L_P、L_F、L、E、R 的顺序打破完全同分。S 只在全微调 N/L 中选，同分取 N。模型推理确定后固定，每次只加载一条路径。

## 导出与推理

训练完成后，工作区的导出脚本按指定种子收集十四库各自的 C 权重，生成带 SHA-256 的 Predictor API 1.0 目录。该工具每次导出一套逐库 checkpoint。三个种子如何交付，按官方说明确定，不根据跨种子得分选取要交付的种子。导出使用正式训练时的同一份配置文件，其相对数据路径在复现环境中重新挂载，配置中不存放凭证。

```bash
python semifinal/export_submission.py --config train_config.json --runs runs \
  --seed "$SEED" --out player_package
```

`predict.py` 使用赛事的 `dataset_name` 选择 checkpoint，核对通道数、窗点数和标签列顺序，返回有限的 `float32 [B,K]` 分类分数。推理在评估模式下按最多 4 窗分块前向，再按原顺序拼接，以控制大批量和高通道任务的显存峰值。它只使用本包所带的权重与训练统计，以及当前窗口数据。交付入口、完整包要求与预检方式见[提交包规范](../public_evaluation/SUBMISSION_SPEC.md)和[复赛材料规则](../public_evaluation/复赛材料提交规则.md)。
