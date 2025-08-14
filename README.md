# VLM 公式识别与评估框架

<p align="center">
  <a href="https://www.modelscope.cn/datasets/JimmyMa99/VLM-formula-recognition-dataset_intern_camp">
    <img alt="ModelScope Dataset" src="https://img.shields.io/badge/ModelScope-Dataset-orange.svg"/>
  </a>
</p>
本项目是一个用于评估视觉语言模型（VLM）在数学公式识别（Formula Recognition）任务上性能的框架。它提供了一套完整的从模型推理到性能评估的工具，可以帮助开发者快速验证模型的效果。

## 项目概述

该框架主要包含两大核心模块：

1.  **推理模块 (`infer_core`)**: 使用指定的VLM（例如 `OpenGVLab/InternVL3-1B`）对输入的公式图片进行识别，并生成对应的 LaTeX 格式文本。
2.  **评估模块 (`eval_core`)**: 对模型生成的 LaTeX 文本进行全面的性能评估。评估过程会将 LaTeX 重新渲染为图片，并从两个维度与原始的参考图片进行比较：
    *   **哈希值比对**: 一种严格的、像素级别的精确匹配。
    *   **图像相似度**: 一种更灵活的、考虑了细微渲染差异的匹配。

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
│   ├── inferVLM.py           # VLM模型推理脚本
│   └── ...
├── main_eval.py              # 主评估脚本
├── env.sh                    # 环境配置脚本
└── README.md                 # 项目说明
```


## 使用方法

### 1. 环境准备

请确保已安装所有必需的 Python 库。您可以根据 `import` 语句安装相关依赖：

```bash
bash env.sh

pip install -r requirements.txt
```

### 2. 模型推理

您可以使用 `infer_core/inferVLM.py` 脚本来对单个或批量图片进行公式识别。

**运行示例:**

在 `infer_core/inferVLM.py` 脚本的 `if __name__ == "__main__":` 部分，修改以下路径：

- `model_path`: 指向您本地的VLM模型权重目录。
- `input_directory`: 包含待识别图片 (`.png` 格式) 的目录。
- `output_directory`: 用于保存生成的 LaTeX 文本 (`.txt` 格式) 的目录。

然后直接运行脚本：

```bash
python infer_core/inferVLM.py
```

脚本将处理输入目录中的所有 `sample*.png` 文件，并将识别结果保存到输出目录。

### 3. 性能评估

评估是对模型生成的 LaTeX 文本的准确性进行打分。

**运行示例:**

在 `main_eval.py` 脚本的 `main()` 函数中，配置以下参数：

- `txt_dir`: 包含模型生成的 LaTeX 文本文件 (`.txt`) 的目录 (即上一步的 `output_directory`)。
- `ref_dir`: 包含原始参考图片 (`.png`) 的目录 (例如 `data/output_eval`)。
- `report_path`: 评估报告的输出路径。
- `hash_weight` 和 `similarity_weight`: 哈希比较和相似度计算在最终得分中的权重。
- `similarity_threshold`: 判定相似度计算是否通过的阈值。

配置完成后，运行主评估脚本：

```bash
python main_eval.py
```

脚本将执行综合评估，并在控制台打印详细的评估结果，同时生成一份评估报告。

## 评估指标

- **哈希比较成功率**: 生成的图片与参考图片哈希值完全相同的样本比例。
- **相似度比较成功率**: 生成的图片与参考图片的综合相似度得分高于设定阈值的样本比例。
- **最终综合得分**: 上述两项成功率的加权平均值，全面反映模型性能。
