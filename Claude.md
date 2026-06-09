# Claude.md

## 1. 项目目标

本项目是课程论文配套科研代码项目，论文题目为：

**Research on Low-Bit (INT4) Post-Training Quantization for Lightweight CNNs Based on Activation Outlier Suppression**

核心目标是在 CIFAR-10 分类任务上，对 compact CNN 或 CIFAR-style ResNet-18 实现并验证 INT4 post-training quantization (PTQ)，重点研究 activation outliers 对低比特量化的影响，以及 layer-wise activation clipping 是否能降低量化误差并缓解精度退化。

必须比较以下实验组：

| Method | Purpose |
|---|---|
| FP32 | 浮点基线，上界参考 |
| INT8-MinMax | 量化流程 sanity check |
| INT4-MinMax | 主 baseline |
| INT4-P99.9 fixed clipping | 固定 percentile clipping baseline |
| INT4-MSE-Selected activation clipping | 本项目 proposed method |

核心方法为 **Layer-wise MSE-Selected Activation Clipping**：

- 对每一层 activation，在 calibration set 上收集浮点激活值。
- 从候选 percentile threshold 集合 `{99.0, 99.5, 99.9, 99.95, 100.0}` 中生成候选 clipping threshold。
- 对每个候选 threshold 进行 fake quantization / dequantization。
- 计算原始 activation 与重构 activation 的 MSE。
- 每层选择 MSE 最小的 threshold 作为该层 activation clipping 上界。

本项目优先保证一个完整、可信、可解释的实验闭环，而不是追求 SOTA。

## 2. 不做什么

本项目必须避免范围膨胀。除非用户明确批准，不做以下内容：

- 不做 Quantization-Aware Training (QAT)。
- 不重新设计大型量化框架。
- 不实现真实 INT4 kernel。
- 不声称真实硬件 latency speedup。
- 不引入 ImageNet、COCO、YOLO 等大规模任务。
- 不把 MobileNetV2 作为必须完成项；它只能作为 optional experiment。
- 不承诺 proposed method 接近 INT8 accuracy 或达到 SOTA performance。
- 不把 MSE-Selected 写成无条件全局最优；如使用 optimal，必须限定为 optimal within a predefined candidate threshold set。
- 不做与课程论文主线无关的复杂工程化重构。

## 3. 技术栈

推荐技术栈如下：

- Language: Python 3.10+
- Deep learning: PyTorch, torchvision
- Dataset: CIFAR-10
- Models: CIFAR-style ResNet-18 或 compact CNN
- Quantization: custom simulated PTQ / fake quantization
- Numerical tools: NumPy
- Experiment logging: CSV / JSON / plain text logs
- Plotting: Matplotlib, optional Seaborn
- Config: YAML 或 argparse，优先保持简单
- Testing: pytest 或最小化 smoke tests
- Environment: CPU 可运行，CUDA 可选

除非确有必要，不引入 TensorRT、ONNX Runtime、TVM、custom CUDA extension 或大型实验管理框架。

## 4. 目录结构

项目采用 **src 包裹布局**。核心代码放在 `src/` 下；保留少量顶层辅助目录 `configs/`、`data/`、`checkpoints/`。项目边界以论文实验闭环为中心，避免模块之间互相偷职责。

```text
project-root/
  Claude.md
  README.md
  requirements.txt
  configs/
  data/
  checkpoints/
  src/
    models/
    quant/
    experiments/
    utils/
  outputs/
    results/
    figures/
    logs/
  paper/
```

后续新增文件时，必须先确认目标文件属于哪个模块。若一个文件看起来同时属于多个模块，应优先拆分职责，而不是创建跨边界的混合文件。

## 5. 模块职责

`src/models/`

- 只放模型结构：compact CNN、CIFAR-style ResNet-18，以及必要的 model factory。
- 对外提供模型构建入口，例如 `build_model(name, num_classes=10)`。
- 不写训练、评估、量化、数据加载、实验流程逻辑。
- 可以依赖 `src/utils/` 中无业务假设的通用工具。
- 禁止依赖 `src/quant/` 或 `src/experiments/`。

`src/quant/`

- 只放量化逻辑：fake quant、observer、calibration、INT8/INT4 MinMax、P99.9 clipping、MSE-Selected clipping。
- 对外提供量化入口，例如 calibration、fake quant、threshold selection、PTQ wrapper。
- 不负责跑完整实验，不保存论文结果表，不生成论文图。
- 可以依赖 `src/utils/`。
- 禁止依赖 `src/experiments/`。

`src/experiments/`

- 只负责实验流程编排：调用 model、data、quant、metrics，跑 FP32、INT8-MinMax、INT4-MinMax、INT4-P99.9、INT4-MSE-Selected。
- 对外提供命令行或函数入口，例如 `run_ptq_experiment(config)`。
- 统一把实验结果写入 `outputs/`。
- 不实现量化公式，不定义模型结构，不把通用工具写在实验脚本里。

`src/utils/`

- 只放通用工具：seed、device、metrics、logging、CSV/JSON I/O、路径处理。
- 不放项目核心算法。
- 不反向依赖 `src/models/`、`src/quant/`、`src/experiments/`。

`outputs/`

- 只存实验产物。
- `outputs/results/` 保存 CSV / JSON 结果。
- `outputs/figures/` 保存实验图和论文候选图。
- `outputs/logs/` 保存运行日志。
- 代码不得从 `outputs/` 导入模块。

`paper/`

- 只写论文：LaTeX/Markdown、参考文献、论文用图表副本。
- 不放实验运行脚本。
- 论文表述必须与代码中的实验设置一致。

`configs/`

- 保存实验配置。
- 不承担实验执行逻辑。

`data/`

- 用作 CIFAR-10 本地缓存。
- 不提交大型数据文件。

`checkpoints/`

- 用于保存或读取模型权重。
- 不视为实验结果表。

依赖方向固定为：

```text
experiments -> models / quant / utils
quant       -> utils
models      -> utils
utils       -> no project modules
```

禁止依赖方向：

```text
models -> quant
quant -> experiments
utils -> models / quant / experiments
outputs -> imported by code
paper -> imported by code
```

## 6. 实验流程

标准实验流程如下：

1. 固定随机种子，准备 CIFAR-10 dataloader。
2. 获得 FP32 baseline。
3. 从 CIFAR-10 training set 中抽取 calibration set。
4. 对 FP32 模型运行 calibration，收集 activation statistics。
5. 运行 INT8-MinMax PTQ，验证量化流程基本正确。
6. 运行 INT4-MinMax PTQ，得到主 baseline。
7. 运行 INT4-P99.9 fixed clipping，所有层使用 P99.9。
8. 运行 INT4-MSE-Selected activation clipping，每层从候选集合中选择 MSE 最小 threshold。
9. 在完整 CIFAR-10 test set 上评估所有方法。
10. 输出主结果表：Top-1 accuracy 与 accuracy drop vs FP32。
11. 输出分析图：activation histogram 与 layer-wise activation MSE。
12. 若 accuracy 提升有限，重点分析 activation MSE、logit consistency 和 layer sensitivity。

实验报告必须说明：

- threshold selection 使用 calibration set；
- classification accuracy 使用 CIFAR-10 test set；
- 真实硬件 latency 未测试；
- compression / storage reduction 只能作为理论估计。

## 7. Git 提交规范

提交必须小而清晰，每个 commit 只做一类事情。

推荐 commit message 格式：

```text
<type>: <short description>
```

可用 type：

- `docs`: 文档、论文草稿、说明文件
- `data`: 数据加载或数据划分
- `model`: 模型结构或 checkpoint 相关
- `quant`: 量化、observer、calibration、clipping
- `exp`: 实验脚本、结果记录、图表生成
- `test`: 测试或 smoke check
- `fix`: bug 修复
- `refactor`: 不改变行为的结构调整

示例：

```text
docs: add project coding rules
quant: implement int4 fake quantization
exp: add cifar10 ptq comparison script
```

提交前必须确认：

- 没有提交大型数据集文件。
- 没有提交无关临时文件。
- 没有混入与当前任务无关的重构。
- 结果文件若提交，必须能说明来源和实验配置。

## 8. AI 写代码时必须遵守的规则

AI 助手在本项目中必须遵守以下规则：

- 修改前必须先给 plan。
- 修改后必须给验证方式。
- 未经用户确认，不得擅自扩大实验范围。
- 优先实现最小可运行闭环，再添加 optional analysis。
- 代码必须服务于课程论文，不做炫技式抽象。
- 所有实验设置必须能在论文中解释清楚。
- 所有指标命名必须稳定，例如 `top1_accuracy`、`accuracy_drop`、`activation_mse`、`logit_mse`。
- INT4 / INT8 量化范围必须写清楚，不允许魔法数字散落在代码中。
- Calibration set 和 test set 必须严格分开。
- Proposed method 和 fixed percentile baseline 必须严格区分。
- 如果实验结果不理想，不允许硬编结论，应改为分析 layer sensitivity、error accumulation 和 limitation。
- 不允许声称真实部署加速，除非真的实现并测量真实 INT4 kernel。
- 不允许删除或覆盖用户已有文件，除非用户明确要求。
- 不允许在没有解释的情况下引入大型依赖。
- 每个脚本应有明确输入、输出和复现实验的方法。

## 9. 每次修改前必须先给 plan

任何文件修改前，AI 必须先给出简短 plan。Plan 至少包含：

- 本次要修改哪些文件。
- 为什么要修改这些文件。
- 本次不会修改哪些范围。
- 预计如何验证修改是否正确。

推荐格式：

```text
Plan:
1. 修改 ...
2. 保持 ... 不变
3. 使用 ... 验证
```

如果用户只是在讨论思路、论文结构或实验设计，AI 不应直接修改文件。

## 10. 每次修改后必须给验证方式

每次完成修改后，AI 必须说明验证方式。验证方式应根据修改类型选择：

- 文档修改：说明检查了结构、术语一致性、是否覆盖用户要求。
- 代码修改：给出实际运行的命令和结果摘要。
- 实验脚本修改：给出 smoke test 或小规模运行命令。
- 量化逻辑修改：至少验证 tensor shape、bit range、scale / zero-point、MSE 输出。
- 结果或图表修改：说明数据来源、生成命令和输出路径。

推荐格式：

```text
Verification:
- Ran ...
- Checked ...
- Output ...
```

如果因为环境、算力或依赖原因无法验证，必须明确说明未验证的原因，并给出用户可运行的验证命令。
