import os
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import glob
from pathlib import Path
from tqdm import tqdm
import time

class MathFormulaOCR:
    def __init__(self, model_path='OpenGVLab/InternVL3-1B', load_in_8bit=False):
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
        
    def build_transform(self, input_size):
        """构建图像预处理变换"""
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """找到最接近的宽高比"""
        best_ratio_diff = float('inf')
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

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """动态预处理图像"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
                ((i // (target_width // image_size)) + 1) * image_size
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
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
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

    def split_model(self):
        """分割模型到多个GPU"""
        device_map = {}
        world_size = torch.cuda.device_count()
        
        if world_size <= 1:
            return 'auto'
            
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
                
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    def load_model(self):
        """加载模型和分词器"""
        print("正在加载模型...")
        device_map = self.split_model()
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=self.load_in_8bit,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            use_fast=False
        )
        print("模型加载完成!")

    def inference_single_image(self, image_path, prompt="请根据图片中的公式生成对应的 latex 公式文本", max_num=12):
        """对单张图片进行推理"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用load_model()方法")
            
        # 加载图像
        pixel_values = self.load_image(image_path, max_num=max_num).to(torch.bfloat16).cuda()
        
        # 构建问题
        question = f'<image>\n{prompt}'
        
        # 生成配置
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        
        # 推理
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        
        return response

    def inference_batch_images(self, image_files, prompt="请根据图片中的公式生成对应的 latex 公式文本", max_num=12):
        """批量推理图片"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用load_model()方法")
        
        # 批量加载图像
        pixel_values, num_patches_list = self.load_images_batch(image_files, max_num=max_num)
        
        if pixel_values.size(0) == 0:
            return []
            
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        # 构建问题列表
        questions = [f'<image>\n{prompt}'] * len(image_files)
        
        # 生成配置
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        
        # 批量推理
        responses = self.model.batch_chat(
            self.tokenizer, 
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config
        )
        
        return responses

    def process_images_batch(self, input_dir, output_dir, batch_size=4, prompt="请根据图片中的公式生成对应的 latex 公式文本"):
        """批量处理图片，支持batch推理"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        # 使用tqdm显示进度条
        with tqdm(total=len(image_files), desc="处理进度", unit="图片") as pbar:
            # 分批处理
            for batch_idx in range(0, len(image_files), batch_size):
                batch_files = image_files[batch_idx:batch_idx + batch_size]
                batch_start_time = time.time()
                
                try:
                    # 批量推理
                    responses = self.inference_batch_images([str(f) for f in batch_files], prompt)
                    
                    # 保存结果
                    for image_file, response in zip(batch_files, responses):
                        output_filename = image_file.stem + '.txt'
                        output_path = Path(output_dir) / output_filename
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(response)
                    
                    processed_count += len(batch_files)
                    batch_time = time.time() - batch_start_time
                    
                    # 更新进度条
                    pbar.update(len(batch_files))
                    
                    # 计算并显示时间估算
                    if processed_count > 0:
                        avg_time_per_batch = (time.time() - start_time) / (batch_idx // batch_size + 1)
                        remaining_batches = total_batches - (batch_idx // batch_size + 1)
                        estimated_remaining_time = avg_time_per_batch * remaining_batches
                        
                        pbar.set_postfix({
                            'batch_time': f'{batch_time:.2f}s',
                            'avg_time': f'{avg_time_per_batch:.2f}s/batch',
                            'ETA': f'{estimated_remaining_time:.0f}s'
                        })
                    
                except Exception as e:
                    print(f"\n处理batch {batch_idx//batch_size + 1}时出错: {str(e)}")
                    # 如果batch推理失败，回退到单张处理
                    for image_file in batch_files:
                        try:
                            response = self.inference_single_image(str(image_file), prompt)
                            output_filename = image_file.stem + '.txt'
                            output_path = Path(output_dir) / output_filename
                            
                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(response)
                                
                            pbar.update(1)
                            print(f"单独处理完成: {image_file.name}")
                            
                        except Exception as single_e:
                            print(f"处理{image_file.name}失败: {str(single_e)}")
                            pbar.update(1)
                            continue
        
        total_time = time.time() - start_time
        print(f"\n批量处理完成！总耗时: {total_time:.2f}秒")
        print(f"平均每张图片耗时: {total_time/len(image_files):.2f}秒")

    def process_single_image_with_save(self, image_path, output_path, prompt="请根据图片中的公式生成对应的 latex 公式文本"):
        """处理单张图片并保存结果"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        try:
            # 推理
            response = self.inference_single_image(image_path, prompt)
            
            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response)
                
            print(f"结果已保存到: {output_path}")
            return response
            
        except Exception as e:
            print(f"处理图片时出错: {str(e)}")
            return None


# 使用示例
if __name__ == "__main__":
    # 初始化OCR类
    ocr = MathFormulaOCR(model_path='/root/code/camp6/swift_output/SFT-InternVL3-1B-lora/v4-20250814-003759/checkpoint-7095-merged', load_in_8bit=False)
    
    # 加载模型
    ocr.load_model()
    
    # 批量处理图片
    input_directory = "Aeval_mini/enhanced_formulas"  # 包含sample*.png文件的目录
    output_directory = "/root/code/camp6/swift_output/SFT-InternVL3-1B-lora/v4-20250814-003759/checkpoint-7095-merged/results_batch_mini"  # 输出txt文件的目录
    
    # 自定义提示词
    custom_prompt = "请根据图片中的公式生成对应的 latex 公式文本"
    
    # 批量处理，设置batch_size（根据显存大小调整，建议从2-8开始尝试）
    batch_size = 16  # 可以根据显存大小调整
    ocr.process_images_batch(input_directory, output_directory, batch_size=batch_size, prompt=custom_prompt)
    
    # 或者处理单张图片
    # single_image_path = "./images/sample1.png"
    # single_output_path = "./results/sample1.txt"
    # result = ocr.process_single_image_with_save(single_image_path, single_output_path, custom_prompt)
    # print(f"识别结果: {result}")