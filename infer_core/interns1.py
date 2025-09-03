import os
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, AutoModelForCausalLM
import glob
from pathlib import Path


class MathFormulaOCRInternS1:
    def __init__(self, model_path="internlm/Intern-S1-mini", load_in_8bit=False):
        """
        初始化数学公式OCR类，专为InternS1模型设计

        Args:
            model_path: 模型路径
            load_in_8bit: 是否使用8bit量化加载
        """
        self.model_path = model_path
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.processor = None
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

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

    def load_model(self):
        """加载模型和处理器"""
        print("正在加载模型...")

        # 使用正确的API加载模型
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        ).eval()

        print("模型加载完成!")

    def inference_single_image(
        self,
        image_path,
        prompt="请根据图片中的公式生成对应的 latex 公式文本",
        max_num=12,
    ):
        """对单张图片进行推理"""
        if self.model is None or self.processor is None:
            raise ValueError("模型未加载，请先调用load_model()方法")

        # 加载图像（与 testimg.py 对齐：直接将 PIL Image 交给消息模板）
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 使用官方卡片推荐的 chat 模板来构造输入
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 在部分 transformers 版本中，BatchEncoding.to 不支持 dtype 参数。
        # 为保持与模型精度一致，这里仅在存在 pixel_values 时手动转换其 dtype。
        if "pixel_values" in inputs:
            try:
                model_dtype = next(self.model.parameters()).dtype
                if isinstance(inputs["pixel_values"], torch.Tensor):
                    inputs["pixel_values"] = inputs["pixel_values"].to(
                        dtype=model_dtype
                    )
            except StopIteration:
                pass

        # 生成（对齐 testimg.py 推荐超参）
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            temperature=0.8,
        )

        # 解码，仅保留新生成部分
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return decoded_output

    def process_images_batch(
        self,
        input_dir,
        output_dir,
        prompt="请根据图片中的公式生成对应的 latex 公式文本",
    ):
        """批量处理图片"""
        if self.model is None or self.processor is None:
            self.load_model()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 查找所有sample*.png文件
        input_path = Path(input_dir)
        image_files = sorted(input_path.glob("sample*.png"))

        if not image_files:
            print(f"在{input_dir}中没有找到sample*.png文件")
            return

        print(f"找到{len(image_files)}个图片文件")

        for image_file in image_files:
            try:
                print(f"正在处理: {image_file.name}")

                # 推理
                response = self.inference_single_image(str(image_file), prompt)

                # 生成输出文件名
                output_filename = image_file.stem + ".txt"  # sample1.png -> sample1.txt
                output_path = Path(output_dir) / output_filename

                # 保存结果
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response)

                print(f"结果已保存到: {output_path}")

            except Exception as e:
                print(f"处理{image_file.name}时出错: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

    def process_single_image_with_save(
        self,
        image_path,
        output_path,
        prompt="请根据图片中的公式生成对应的 latex 公式文本",
    ):
        """处理单张图片并保存结果"""
        if self.model is None or self.processor is None:
            self.load_model()

        try:
            # 推理
            response = self.inference_single_image(image_path, prompt)

            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存结果
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response)

            print(f"结果已保存到: {output_path}")
            return response

        except Exception as e:
            print(f"处理图片时出错: {str(e)}")
            import traceback

            traceback.print_exc()
            return None
