import os
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer
import glob
from pathlib import Path
from tqdm import tqdm
import time
import re


class MathFormulaOCR_vl:
    def __init__(self, model_path="OpenGVLab/InternVL3_5-8B", load_in_8bit=False):
        """
        初始化数学公式OCR类

        Args:
            model_path: 模型路径
            load_in_8bit: 是否使用8bit量化加载
        """
        self.model_path = model_path
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.tokenizer = None
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

    def extract_latex_formula(self, text):
        """
        从模型输出中提取最后一个latex代码块，简化提取逻辑
        """
        if not text:
            return ""

        # 先去掉@符号及之前所有内容
        at_index = text.find("</think>")
        if at_index != -1:
            text = text[at_index + 8 :].strip()

        # 只提取最后一个```latex代码块
        latex_block_pattern = r"```latex\s*\n(.*?)\n```"
        matches = re.findall(latex_block_pattern, text, re.DOTALL)

        if matches:
            # 返回最后一个匹配的LaTeX代码块
            return matches[-1].strip()

        # 如果没有找到```latex代码块，尝试找```代码块（不指定语言）
        code_block_pattern = r"```\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)

        if matches:
            # 返回最后一个代码块
            return matches[-1].strip()

        # 如果都没找到，返回原文本
        return text.strip()

    def clean_latex_formula(self, formula):
        """
        简单清理LaTeX公式
        """
        if not formula:
            return ""

        # 只做基本清理：去除首尾空白
        return formula.strip()

    def build_transform(self, input_size):
        """构建图像预处理变换"""
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        """找到最接近的宽高比"""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        """动态预处理图像"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        """加载并预处理图像"""
        image = Image.open(image_file).convert("RGB")
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def load_images_batch(self, image_files, input_size=448, max_num=12):
        """批量加载并预处理图像"""
        all_pixel_values = []
        num_patches_list = []

        for image_file in image_files:
            pixel_values = self.load_image(image_file, input_size, max_num)
            all_pixel_values.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))

        # 将所有图像拼接到一个tensor中
        if all_pixel_values:
            pixel_values = torch.cat(all_pixel_values, dim=0)
        else:
            pixel_values = torch.empty(0)

        return pixel_values, num_patches_list

    def load_model(self):
        """加载模型和分词器"""
        print("正在加载模型...")

        # 使用官方推荐的方式加载模型
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=self.load_in_8bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",  # 使用auto device map
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        print("模型加载完成!")

    def inference_single_image(
        self,
        image_path,
        prompt="请根据图片中的公式生成对应的 latex 公式文本",
        max_num=12,
    ):
        """对单张图片进行推理，返回(提取的公式, 完整响应)"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用load_model()方法")

        # 加载图像
        pixel_values = (
            self.load_image(image_path, max_num=max_num).to(torch.bfloat16).cuda()
        )

        # 构建问题
        question = f"<image>\n{prompt}"

        # 生成配置
        generation_config = dict(max_new_tokens=32768, do_sample=True)

        # 推理
        response = self.model.chat(
            self.tokenizer, pixel_values, question, generation_config
        )

        # 提取并清理LaTeX公式
        latex_formula = self.extract_latex_formula(response)
        cleaned_formula = self.clean_latex_formula(latex_formula)

        return cleaned_formula, response

    def inference_batch_images(
        self,
        image_files,
        prompt="请根据图片中的公式生成对应的 latex 公式文本",
        max_num=12,
    ):
        """批量推理图片，返回(提取的公式列表, 完整响应列表)"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用load_model()方法")

        # 批量加载图像
        pixel_values, num_patches_list = self.load_images_batch(
            image_files, max_num=max_num
        )

        if pixel_values.size(0) == 0:
            return [], []

        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # 构建问题列表 - 每个图像对应一个问题
        questions = [f"<image>\n{prompt}"] * len(image_files)

        # 生成配置
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        # 批量推理 - 使用官方的batch_chat方法
        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )

        # 批量提取并清理LaTeX公式
        cleaned_responses = []
        for response in responses:
            latex_formula = self.extract_latex_formula(response)
            cleaned_formula = self.clean_latex_formula(latex_formula)
            cleaned_responses.append(cleaned_formula)

        return cleaned_responses, responses

    def process_images_batch(self, input_dir, output_dir, prompt, batch_size=4):
        """批量处理图片，支持batch推理，同时保存提取的公式和完整响应"""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 创建存放完整响应的子目录
        result_text_dir = os.path.join(output_dir, "result_text")
        os.makedirs(result_text_dir, exist_ok=True)

        # 查找所有sample*.png文件
        input_path = Path(input_dir)
        image_files = sorted(input_path.glob("sample*.png"))

        if not image_files:
            print(f"在{input_dir}中没有找到sample*.png文件")
            return

        print(f"找到{len(image_files)}个图片文件，使用batch_size={batch_size}")

        # 计算总批次数
        total_batches = math.ceil(len(image_files) / batch_size)

        # 用于时间估算
        start_time = time.time()
        processed_count = 0
        successful_count = 0

        # 使用tqdm显示进度条
        with tqdm(total=len(image_files), desc="处理进度", unit="图片") as pbar:
            # 分批处理
            for batch_idx in range(0, len(image_files), batch_size):
                batch_files = image_files[batch_idx : batch_idx + batch_size]
                batch_start_time = time.time()

                try:
                    # 批量推理，获取提取的公式和完整响应
                    extracted_formulas, full_responses = self.inference_batch_images(
                        [str(f) for f in batch_files], prompt
                    )

                    # 保存结果
                    for image_file, extracted_formula, full_response in zip(
                        batch_files, extracted_formulas, full_responses
                    ):
                        base_filename = image_file.stem

                        # 保存提取的公式
                        output_path = Path(output_dir) / f"{base_filename}.txt"
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(extracted_formula)

                        # 保存完整响应
                        result_text_path = (
                            Path(result_text_dir) / f"{base_filename}.txt"
                        )
                        with open(result_text_path, "w", encoding="utf-8") as f:
                            f.write(full_response)

                        successful_count += 1
                        print(
                            f"提取的公式: {extracted_formula[:100]}{'...' if len(extracted_formula) > 100 else ''}"
                        )

                    processed_count += len(batch_files)
                    batch_time = time.time() - batch_start_time

                    # 更新进度条
                    pbar.update(len(batch_files))

                    # 计算并显示时间估算
                    if processed_count > 0:
                        avg_time_per_batch = (time.time() - start_time) / (
                            batch_idx // batch_size + 1
                        )
                        remaining_batches = total_batches - (
                            batch_idx // batch_size + 1
                        )
                        estimated_remaining_time = (
                            avg_time_per_batch * remaining_batches
                        )

                        pbar.set_postfix(
                            {
                                "batch_time": f"{batch_time:.2f}s",
                                "avg_time": f"{avg_time_per_batch:.2f}s/batch",
                                "ETA": f"{estimated_remaining_time:.0f}s",
                            }
                        )

                except Exception as e:
                    print(f"\n处理batch {batch_idx // batch_size + 1}时出错: {str(e)}")
                    # 如果batch推理失败，回退到单张处理
                    for image_file in batch_files:
                        try:
                            extracted_formula, full_response = (
                                self.inference_single_image(str(image_file), prompt)
                            )
                            base_filename = image_file.stem

                            # 保存提取的公式
                            output_path = Path(output_dir) / f"{base_filename}.txt"
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write(extracted_formula)

                            # 保存完整响应
                            result_text_path = (
                                Path(result_text_dir) / f"{base_filename}.txt"
                            )
                            with open(result_text_path, "w", encoding="utf-8") as f:
                                f.write(full_response)

                            successful_count += 1
                            pbar.update(1)
                            print(f"单独处理完成: {image_file.name}")
                            print(
                                f"提取的公式: {extracted_formula[:100]}{'...' if len(extracted_formula) > 100 else ''}"
                            )

                        except Exception as single_e:
                            print(f"处理{image_file.name}失败: {str(single_e)}")
                            pbar.update(1)
                            continue

                # 定期清理GPU缓存
                if (batch_idx // batch_size + 1) % 5 == 0:
                    torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"\n批量处理完成！")
        print(f"成功处理: {successful_count}/{len(image_files)} 个文件")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"提取的公式保存在: {output_dir}")
        print(f"完整响应保存在: {result_text_dir}")
        if successful_count > 0:
            print(f"平均每张图片耗时: {total_time / successful_count:.2f}秒")

    def process_single_image_with_save(
        self,
        image_path,
        output_path,
        prompt="请根据图片中的公式生成对应的 latex 公式文本",
    ):
        """处理单张图片并保存结果，同时保存提取的公式和完整响应"""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            # 推理，获取提取的公式和完整响应
            extracted_formula, full_response = self.inference_single_image(
                image_path, prompt
            )

            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存提取的公式
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(extracted_formula)

            # 保存完整响应（在同一目录下创建result_text子目录）
            output_dir = os.path.dirname(output_path)
            result_text_dir = os.path.join(output_dir, "result_text")
            os.makedirs(result_text_dir, exist_ok=True)

            base_filename = os.path.splitext(os.path.basename(output_path))[0]
            result_text_path = os.path.join(result_text_dir, f"{base_filename}.txt")

            with open(result_text_path, "w", encoding="utf-8") as f:
                f.write(full_response)

            print(f"提取的公式已保存到: {output_path}")
            print(f"完整响应已保存到: {result_text_path}")
            print(
                f"提取的公式: {extracted_formula[:100]}{'...' if len(extracted_formula) > 100 else ''}"
            )
            return extracted_formula, full_response

        except Exception as e:
            print(f"处理图片时出错: {str(e)}")
            return None, None

    def test_extraction(self, test_text):
        """测试提取功能"""
        print("原始文本:")
        print(test_text)
        print("\n提取的LaTeX公式:")
        extracted = self.extract_latex_formula(test_text)
        cleaned = self.clean_latex_formula(extracted)
        print(cleaned)
        return cleaned
