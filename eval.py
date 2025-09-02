#!/usr/bin/env python3
"""
一键测评脚本
合并模型推理和性能评估两个步骤，实现完整的公式识别测评流程
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# 添加路径导入
sys.path.append("./infer_core")
sys.path.append("./eval_core")
from infer_core.inferVLM import MathFormulaOCR
from infer_core.interns1 import MathFormulaOCRInternS1
from main_eval import ComprehensiveEvaluator


class EvaluatorPipeline:
    """评估管道，封装完整的推理和评估流程"""

    def __init__(
        self,
        model_path,
        hash_weight=0.5,
        similarity_weight=0.5,
        similarity_threshold=0.6,
    ):
        """
        初始化评估管道

        Args:
            model_path (str): 模型路径
            hash_weight (float): 哈希比较权重
            similarity_weight (float): 相似度计算权重
            similarity_threshold (float): 相似度阈值
        """
        self.model_path = model_path
        self.hash_weight = hash_weight
        self.similarity_weight = similarity_weight
        self.similarity_threshold = similarity_threshold

        # 初始化组件
        self.ocr = None
        self.evaluator = None

        print("=" * 80)
        print("🚀 初始化测评管道")
        print("=" * 80)
        print(f"模型路径: {self.model_path}")
        print(f"哈希权重: {self.hash_weight}")
        print(f"相似度权重: {self.similarity_weight}")
        print(f"相似度阈值: {self.similarity_threshold}")
        print("=" * 80)

    def _init_ocr(self):
        """初始化OCR组件"""
        if self.ocr is None:
            print("📥 正在初始化OCR组件...")
            # 根据模型路径判断使用哪个OCR类
            if "internvl3" or "InternVL3" in self.model_path.lower():
                self.ocr = MathFormulaOCR(
                    model_path=self.model_path, load_in_8bit=False
                )
                print("✅ 使用InternVL3模型ocr组件初始化完成")
            else:
                self.ocr = MathFormulaOCRInternS1(
                    model_path=self.model_path, load_in_8bit=False
                )
                print("✅ 使用InternS1模型ocr组件初始化完成")

    def _init_evaluator(self):
        """初始化评估器组件"""
        if self.evaluator is None:
            print("📊 正在初始化评估器组件...")
            self.evaluator = ComprehensiveEvaluator(
                hash_weight=self.hash_weight,
                similarity_weight=self.similarity_weight,
                similarity_threshold=self.similarity_threshold,
            )
            print("✅ 评估器组件初始化完成")

    def run_inference(self, input_directory, output_directory):
        """
        执行模型推理

        Args:
            input_directory (str): 输入图片目录
            output_directory (str): 输出LaTeX文本目录
        """
        print("\n🔍 步骤1: 模型推理")
        print("-" * 60)

        self._init_ocr()

        # 创建输出目录
        os.makedirs(output_directory, exist_ok=True)

        # 批量处理图片
        self.ocr.process_images_batch(input_directory, output_directory)

        print(f"✅ 推理完成，结果保存至: {output_directory}")

    def run_evaluation(self, txt_dir, ref_dir, output_report, keep_temp_images=False):
        """
        执行性能评估

        Args:
            txt_dir (str): LaTeX文本目录
            ref_dir (str): 参考图片目录
            output_report (str): 输出报告路径
            keep_temp_images (bool): 是否保留临时图片

        Returns:
            dict: 评估结果
        """
        print("\n📈 步骤2: 性能评估")
        print("-" * 60)

        self._init_evaluator()

        # 执行综合评估
        results = self.evaluator.evaluate_comprehensive(
            txt_dir=txt_dir,
            ref_dir=ref_dir,
            output_report=output_report,
            keep_temp_images=keep_temp_images,
        )

        print(f"✅ 评估完成，报告生成: {output_report}")
        return results

    def run_complete_pipeline(
        self, input_directory, output_directory, report_path, keep_temp_images=False
    ):
        """
        运行完整的测评流程

        Args:
            input_directory (str): 输入图片目录（同时作为参考目录）
            output_directory (str): 中间结果LaTeX文本目录
            report_path (str): 最终报告路径
            keep_temp_images (bool): 是否保留临时图片

        Returns:
            dict: 最终评估结果
        """
        print("🎯 开始完整测评流程")

        try:
            # 步骤1: 模型推理
            self.run_inference(input_directory, output_directory)

            # 步骤2: 性能评估（使用输入目录作为参考目录）
            print(f"📝 使用输入目录作为参考目录进行评估: {input_directory}")
            results = self.run_evaluation(
                output_directory, input_directory, report_path, keep_temp_images
            )

            print("\n🎉 测评流程全部完成!")
            print(f"📊 最终得分: {results['final_score']:.2f}")

            return results

        except Exception as e:
            print(f"\n❌ 测评流程失败: {e}")
            raise


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="一键公式识别测评脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval.py \\
      --model-path /path/to/model \\
      --input-dir /path/to/images \\
      --output-dir ./results \\
      --report-path ./evaluation_report.txt \\
      --ref-dir ./data/output_eval

  python eval.py \\
      --model-path OpenGVLab/InternVL3-1B \\
      --input-dir ./data/samples_test \\
      --output-dir ./inference_results \\
      --report-path ./report.txt \\
      --ref-dir ./data/output_eval \\
      --hash-weight 0.6 \\
      --similarity-weight 0.4
        """,
    )

    # 必需参数
    parser.add_argument("--model-path", required=True, help="模型路径 (必需)")

    parser.add_argument("--input-dir", required=True, help="输入图片目录 (必需)")

    parser.add_argument(
        "--output-dir", required=True, help="中间结果LaTeX文本输出目录 (必需)"
    )

    parser.add_argument("--report-path", required=True, help="最终评估报告路径 (必需)")

    # 可选参数
    parser.add_argument(
        "--hash-weight", type=float, default=0.5, help="哈希比较权重 (默认: 0.5)"
    )

    parser.add_argument(
        "--similarity-weight",
        type=float,
        default=0.5,
        help="相似度计算权重 (默认: 0.5)",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.99,
        help="相似度阈值 (默认: 0.6)",
    )

    parser.add_argument(
        "--keep-temp-images", action="store_true", help="保留临时生成的图片"
    )

    return parser.parse_args()


def validate_paths(args):
    """验证路径参数"""
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()

        print("🔧 配置信息:")
        print(f"  模型路径: {args.model_path}")
        print(f"  输入目录: {args.input_dir}")
        print(f"  中间输出目录: {args.output_dir}")
        print(f"  报告路径: {args.report_path}")
        print(f"  哈希权重: {args.hash_weight}")
        print(f"  相似度权重: {args.similarity_weight}")
        print(f"  相似度阈值: {args.similarity_threshold}")
        print(f"  保留临时图片: {args.keep_temp_images}")

        # 验证路径
        validate_paths(args)

        # 创建评估管道
        pipeline = EvaluatorPipeline(
            model_path=args.model_path,
            hash_weight=args.hash_weight,
            similarity_weight=args.similarity_weight,
            similarity_threshold=args.similarity_threshold,
        )

        # 运行完整流程
        results = pipeline.run_complete_pipeline(
            input_directory=args.input_dir,
            output_directory=args.output_dir,
            report_path=args.report_path,
            keep_temp_images=args.keep_temp_images,
        )

        print(f"\n🎉 测评完成! 最终得分: {results['final_score']:.2f}")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
