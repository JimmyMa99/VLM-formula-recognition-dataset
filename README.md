# VLM 公式识别与评估框架

<p align="center">
  <a href="https://www.modelscope.cn/datasets/JimmyMa99/VLM-formula-recognition-dataset_intern_camp">
    <img alt="ModelScope Dataset" src="https://img.shields.io/badge/ModelScope-Dataset-orange.svg"/>
  </a>
</p>
本项目是一个用于评估视觉语言模型（VLM）在数学公式识别（Formula Recognition）任务上性能的框架。它提供了一套完整的从模型推理到性能评估的工具，可以帮助开发者快速验证模型的效果。

## 项目概述

该框架主要包含两大核心模块：

1. **推理模块 (`infer_core`)**: 使用指定的VLM（例如 `OpenGVLab/InternVL3-1B`）对输入的公式图片进行识别，并生成对应的 LaTeX 格式文本。
2. **评估模块 (`eval_core`)**: 对模型生成的 LaTeX 文本进行全面的性能评估。评估过程会将 LaTeX 重新渲染为图片，并从两个维度与原始的参考图片进行比较：
   * **哈希值比对**: 一种严格的、像素级别的精确匹配。
   * **图像相似度**: 一种更灵活的、考虑了细微渲染差异的匹配。

最终，系统会根据这两种方法的成功率，通过加权计算得出一个综合得分，以全面反映模型的性能。

## 项目结构

```
├── data/                     # 示例数据
│   ├── output_eval/          # 用于评估的参考图片 (Ground Truth)
│   └── samples_test/         # 包含模型预测的LaTeX文本文件
├── eval_core/                # 评估核心逻辑
│   ├── cal_score.py          # 图像相似度计算
│   └── cal_score_hash.py     # 哈希值比较
├── infer_core/               # 推理核心逻辑
│   ├── infervl.py            # VL模型推理脚本
│   └── ...
├── swift_config/                       # 模型微调配置文件
│   ├── internvl3.5_1b_train.sh         # internvl3.5微调脚本
│   └── ...
├── eval.py                   # 一键评估入口脚本
├── env.sh                    # 环境配置脚本
└── uoload.py                 # 上传模型到魔搭
└── README.md                 # 项目说明
```

## 使用方法

### 1. 环境准备

请确保已安装所有必需的 Python 库。您可以根据 `import` 语句安装相关依赖：

```bash
conda create -n ms-swift python=3.10 -y
conda activate ms-swift

git clone https://gh.llkk.cc/https://github.com/JimmyMa99/VLM-formula-recognition-dataset.git
cd VLM-formula-recognition-dataset

pip install -r requirements.txt


git clone https://xget.xi-xu.me/gh/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

### 2. 一键评估运行

现在已将推理与评估流程合并，直接运行以下命令即可完成从推理到评估的全流程：

```bash
python eval.py \
  --model-path /root/share/new_models/InternVL3/InternVL3-1B\
  --input-dir ./data/output_eval \
  --output-dir ./data/results_vl \
  --report-path ./evaluation_report.txt \
  --model-type vl
```

脚本会完成：读取输入、运行模型推理、渲染对比并输出评估报告。

## 评估指标

- **哈希比较成功率**: 生成的图片与参考图片哈希值完全相同的样本比例。
- **相似度比较成功率**: 生成的图片与参考图片的综合相似度得分高于设定阈值的样本比例。
- **最终综合得分**: 上述两项成功率的加权平均值，全面反映模型性能。
