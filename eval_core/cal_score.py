import os
import sys
import cv2
import numpy as np
from PIL import Image
import tempfile
import shutil
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
import hashlib

# 导入我们之前写的LaTeX转图片类
from infer_core.latex2img_file import LatexToImage

class ImageSimilarity:
    def __init__(self):
        """
        初始化图像相似度计算类
        """
        pass
    
    def _load_image(self, image_path):
        """
        加载图像文件
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            numpy.ndarray: 图像数组，如果加载失败返回None
        """
        try:
            if isinstance(image_path, str):
                # 从文件路径加载
                if not os.path.exists(image_path):
                    print(f"图像文件不存在: {image_path}")
                    return None
                image = cv2.imread(image_path)
                if image is None:
                    # 尝试用PIL加载
                    pil_image = Image.open(image_path)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                # 假设传入的已经是numpy数组
                image = image_path
                
            return image
        except Exception as e:
            print(f"加载图像失败: {e}")
            return None
    
    def _resize_images(self, img1, img2, target_size=(256, 256)):
        """
        将两张图片调整到相同尺寸
        
        Args:
            img1, img2: 输入图像
            target_size: 目标尺寸
            
        Returns:
            tuple: 调整后的两张图像
        """
        img1_resized = cv2.resize(img1, target_size)
        img2_resized = cv2.resize(img2, target_size)
        return img1_resized, img2_resized
    
    def histogram_similarity(self, image1, image2, method='correlation'):
        """
        基于直方图的相似度计算
        
        Args:
            image1, image2: 图像路径或numpy数组
            method: 比较方法 ('correlation', 'chi_square', 'intersection', 'bhattacharyya')
            
        Returns:
            float: 相似度分数 (0-1)，1表示完全相似
        """
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # 调整图像尺寸
        img1, img2 = self._resize_images(img1, img2)
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 计算直方图
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # 归一化直方图
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        
        # 选择比较方法
        if method == 'correlation':
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        elif method == 'chi_square':
            similarity = 1.0 / (1.0 + cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR))
        elif method == 'intersection':
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        elif method == 'bhattacharyya':
            similarity = 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        return max(0.0, min(1.0, similarity))
    
    def ssim_similarity(self, image1, image2):
        """
        基于结构相似性指数(SSIM)的相似度计算
        
        Args:
            image1, image2: 图像路径或numpy数组
            
        Returns:
            float: SSIM分数 (-1到1)，1表示完全相似
        """
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # 调整图像尺寸
        img1, img2 = self._resize_images(img1, img2)
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 计算SSIM
        similarity_score = ssim(gray1, gray2)
        
        return similarity_score
    
    def mse_similarity(self, image1, image2):
        """
        基于均方误差(MSE)的相似度计算
        
        Args:
            image1, image2: 图像路径或numpy数组
            
        Returns:
            float: 相似度分数 (0-1)，1表示完全相似
        """
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # 调整图像尺寸
        img1, img2 = self._resize_images(img1, img2)
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # 计算MSE
        mse = np.mean((gray1 - gray2) ** 2)
        
        # 转换为相似度 (MSE越小相似度越高)
        max_mse = 255 ** 2  # 最大可能的MSE
        similarity = 1.0 - (mse / max_mse)
        
        return max(0.0, similarity)
    
    def feature_similarity(self, image1, image2, detector='orb'):
        """
        基于特征点的相似度计算
        
        Args:
            image1, image2: 图像路径或numpy数组
            detector: 特征检测器类型 ('orb', 'sift')
            
        Returns:
            float: 相似度分数 (0-1)，1表示完全相似
        """
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # 调整图像尺寸
        img1, img2 = self._resize_images(img1, img2)
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 选择特征检测器
        if detector.lower() == 'orb':
            feature_detector = cv2.ORB_create()
        elif detector.lower() == 'sift':
            feature_detector = cv2.SIFT_create()
        else:
            raise ValueError(f"不支持的特征检测器: {detector}")
        
        # 检测关键点和描述符
        kp1, des1 = feature_detector.detectAndCompute(gray1, None)
        kp2, des2 = feature_detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        # 特征匹配
        if detector.lower() == 'orb':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = matcher.match(des1, des2)
        
        if len(matches) == 0:
            return 0.0
        
        # 计算相似度
        good_matches = [m for m in matches if m.distance < 50]  # 可调整阈值
        similarity = len(good_matches) / max(len(kp1), len(kp2))
        
        return min(1.0, similarity)
    
    def perceptual_hash_similarity(self, image1, image2, hash_size=8):
        """
        基于感知哈希的相似度计算
        
        Args:
            image1, image2: 图像路径或numpy数组
            hash_size: 哈希尺寸
            
        Returns:
            float: 相似度分数 (0-1)，1表示完全相似
        """
        def compute_phash(image):
            # 转换为灰度图并调整尺寸
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (hash_size + 1, hash_size))
            
            # 计算水平梯度
            diff = resized[:, 1:] > resized[:, :-1]
            
            # 转换为哈希值
            return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # 计算感知哈希
        hash1 = compute_phash(img1)
        hash2 = compute_phash(img2)
        
        # 计算汉明距离
        hamming_distance = bin(hash1 ^ hash2).count('1')
        
        # 转换为相似度
        max_distance = hash_size * hash_size
        similarity = 1.0 - (hamming_distance / max_distance)
        
        return similarity
    
    def comprehensive_similarity(self, image1, image2, weights=None):
        """
        综合多种方法的相似度计算
        
        Args:
            image1, image2: 图像路径或numpy数组
            weights: 各方法的权重字典
            
        Returns:
            dict: 包含各种方法结果和综合分数的字典
        """
        if weights is None:
            weights = {
                'histogram': 0.2,
                'ssim': 0.3,
                'mse': 0.2,
                'feature': 0.2,
                'phash': 0.1
            }
        
        results = {}
        
        # 计算各种相似度
        results['histogram'] = self.histogram_similarity(image1, image2)
        results['ssim'] = (self.ssim_similarity(image1, image2) + 1) / 2  # 转换到0-1范围
        results['mse'] = self.mse_similarity(image1, image2)
        results['feature'] = self.feature_similarity(image1, image2)
        results['phash'] = self.perceptual_hash_similarity(image1, image2)
        
        # 计算加权平均
        comprehensive_score = sum(results[method] * weights.get(method, 0) 
                                for method in results.keys())
        
        results['comprehensive'] = comprehensive_score
        
        return results


class LatexSimilarityEvaluator:
    """
    LaTeX公式相似度评估器
    用于比较LaTeX文本生成的图片与标准图片的相似度，并按阈值计算最终得分
    """
    
    def __init__(self, dpi=300, fontsize=12, temp_dir=None, similarity_threshold=0.6):
        """
        初始化评估器
        
        Args:
            dpi (int): 图片DPI
            fontsize (int): 字体大小
            temp_dir (str): 临时文件目录，如果为None则使用系统临时目录
            similarity_threshold (float): 相似度阈值，超过此值算1分，否则算0分
        """
        self.latex_converter = LatexToImage(dpi=dpi, fontsize=fontsize)
        self.similarity_calculator = ImageSimilarity()
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.similarity_threshold = similarity_threshold
        
        # 确保临时目录存在
        os.makedirs(self.temp_dir, exist_ok=True)
        
        print(f"LaTeX相似度评估器初始化完成")
        print(f"临时目录: {self.temp_dir}")
        print(f"相似度阈值: {self.similarity_threshold}")
    
    def set_threshold(self, threshold):
        """
        设置相似度阈值
        
        Args:
            threshold (float): 新的相似度阈值 (0-1)
        """
        if 0 <= threshold <= 1:
            self.similarity_threshold = threshold
            print(f"相似度阈值已更新为: {threshold}")
        else:
            raise ValueError("阈值必须在0-1之间")
    
    def _read_latex_from_txt(self, txt_path):
        """
        从txt文件读取LaTeX公式
        
        Args:
            txt_path (str): txt文件路径
            
        Returns:
            str: LaTeX公式内容，如果读取失败返回None
        """
        try:
            if not os.path.exists(txt_path):
                print(f"错误: txt文件不存在 - {txt_path}")
                return None
                
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                print(f"警告: txt文件为空 - {txt_path}")
                return None
                
            return content
            
        except Exception as e:
            print(f"读取txt文件失败 - {txt_path}: {e}")
            return None
    
    def _generate_temp_image(self, latex_content, base_name="temp_latex"):
        """
        生成临时图片
        
        Args:
            latex_content (str): LaTeX公式内容
            base_name (str): 临时文件基础名称
            
        Returns:
            str: 生成的图片路径，如果生成失败返回None
        """
        try:
            # 生成唯一的临时文件名
            import uuid
            temp_filename = f"{base_name}_{uuid.uuid4().hex[:8]}.png"
            temp_image_path = os.path.join(self.temp_dir, temp_filename)
            
            # 使用LaTeX转换器生成图片
            success = self.latex_converter.latex_to_image(latex_content, temp_image_path)
            
            if success and os.path.exists(temp_image_path):
                return temp_image_path
            else:
                print(f"LaTeX图片生成失败")
                return None
                
        except Exception as e:
            print(f"生成临时图片失败: {e}")
            return None
    
    def evaluate_single(self, txt_path, reference_image_path, cleanup=True):
        """
        评估单个LaTeX文本与参考图片的相似度，返回0或1的得分
        
        Args:
            txt_path (str): 包含LaTeX公式的txt文件路径
            reference_image_path (str): 参考图片路径
            cleanup (bool): 是否清理临时文件
            
        Returns:
            dict: 包含得分和详细信息的字典
        """
        result = {
            'txt_path': txt_path,
            'reference_image': reference_image_path,
            'score': 0,  # 最终得分：0或1
            'similarity_score': 0.0,  # 原始相似度分数
            'threshold': self.similarity_threshold,
            'latex_compile_success': False,
            'similarity_above_threshold': False,
            'generated_image': None,
            'latex_content': None,
            'error': None
        }
        
        try:
            # 1. 检查参考图片是否存在
            if not os.path.exists(reference_image_path):
                result['error'] = f"参考图片不存在: {reference_image_path}"
                return result
            
            # 2. 读取LaTeX内容
            latex_content = self._read_latex_from_txt(txt_path)
            if latex_content is None:
                result['error'] = "读取LaTeX内容失败"
                return result
            
            result['latex_content'] = latex_content
            
            # 3. 尝试生成图片 (LaTeX编译)
            temp_image_path = self._generate_temp_image(latex_content)
            if temp_image_path is None:
                result['error'] = "LaTeX编译失败，无法生成图片"
                result['latex_compile_success'] = False
                # LaTeX编译失败，得分为0
                return result
            
            result['latex_compile_success'] = True
            result['generated_image'] = temp_image_path
            
            # 4. 计算相似度
            similarity_results = self.similarity_calculator.comprehensive_similarity(
                temp_image_path, reference_image_path
            )
            result['similarity_score'] = similarity_results['comprehensive']
            result['detailed_scores'] = similarity_results
            
            # 5. 根据阈值判断最终得分
            if result['similarity_score'] >= self.similarity_threshold:
                result['score'] = 1
                result['similarity_above_threshold'] = True
            else:
                result['score'] = 0
                result['similarity_above_threshold'] = False
            
        except Exception as e:
            result['error'] = f"评估过程中发生错误: {str(e)}"
        
        finally:
            # 6. 清理临时文件
            if cleanup and 'generated_image' in result and result['generated_image']:
                try:
                    if os.path.exists(result['generated_image']):
                        os.remove(result['generated_image'])
                        if not cleanup:  # 如果不清理，更新路径信息
                            result['generated_image'] = None
                except Exception as e:
                    print(f"清理临时文件失败: {e}")
        
        return result
    
    def evaluate_batch(self, txt_dir, reference_dir, output_report=None, cleanup=True):
        """
        批量评估LaTeX文本与参考图片的相似度，计算最终得分百分比
        
        Args:
            txt_dir (str): 包含txt文件的目录
            reference_dir (str): 包含参考图片的目录
            output_report (str): 输出报告文件路径
            cleanup (bool): 是否清理临时文件
            
        Returns:
            dict: 包含最终得分和详细结果的字典
        """
        # 查找所有txt文件
        txt_files = []
        for file in os.listdir(txt_dir):
            if file.endswith('.txt'):
                txt_files.append(os.path.join(txt_dir, file))
        
        if not txt_files:
            print(f"在 {txt_dir} 中没有找到txt文件")
            return {
                'final_score': 0.0,
                'total_samples': 0,
                'passed_samples': 0,
                'results': []
            }
        
        txt_files.sort()
        
        print(f"开始批量评估 {len(txt_files)} 个样本...")
        print(f"相似度阈值: {self.similarity_threshold}")
        
        results = []
        passed_count = 0
        total_count = len(txt_files)
        
        for i, txt_path in enumerate(txt_files, 1):
            txt_basename = os.path.basename(txt_path)
            print(f"\n[{i}/{total_count}] 处理: {txt_basename}")
            
            # 尝试找到对应的参考图片
            txt_name = os.path.splitext(txt_basename)[0]
            
            # 尝试常见的图片扩展名
            reference_image = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                potential_ref = os.path.join(reference_dir, txt_name + ext)
                if os.path.exists(potential_ref):
                    reference_image = potential_ref
                    break
            
            if reference_image is None:
                print(f"警告: 找不到对应的参考图片 - {txt_name}")
                result = {
                    'txt_path': txt_path,
                    'reference_image': None,
                    'score': 0,
                    'error': '找不到对应的参考图片'
                }
                results.append(result)
                continue
            
            # 评估单个样本
            result = self.evaluate_single(txt_path, reference_image, cleanup)
            results.append(result)
            
            # 统计得分
            if result['score'] == 1:
                passed_count += 1
                status = "✓ 通过"
            else:
                status = "✗ 未通过"
            
            # 打印详细信息
            if result['latex_compile_success']:
                print(f"{status} - 相似度: {result['similarity_score']:.4f} (阈值: {self.similarity_threshold})")
            else:
                print(f"{status} - LaTeX编译失败")
                if result['error']:
                    print(f"  错误: {result['error']}")
        
        # 计算最终得分百分比
        final_score_percentage = (passed_count / total_count * 100) if total_count > 0 else 0
        
        evaluation_summary = {
            'final_score': final_score_percentage,
            'total_samples': total_count,
            'passed_samples': passed_count,
            'failed_samples': total_count - passed_count,
            'similarity_threshold': self.similarity_threshold,
            'results': results
        }
        
        # 打印总结
        print("\n" + "="*60)
        print("评估完成")
        print("="*60)
        print(f"总样本数: {total_count}")
        print(f"通过样本数: {passed_count}")
        print(f"失败样本数: {total_count - passed_count}")
        print(f"最终得分: {final_score_percentage:.2f}%")
        print(f"相似度阈值: {self.similarity_threshold}")
        
        # 生成详细报告
        if output_report:
            self._generate_evaluation_report(evaluation_summary, output_report)
        
        return evaluation_summary
    
    def _generate_evaluation_report(self, evaluation_summary, report_path):
        """
        生成评估报告
        
        Args:
            evaluation_summary (dict): 评估总结
            report_path (str): 报告文件路径
        """
        try:
            results = evaluation_summary['results']
            
            # 统计LaTeX编译失败和相似度不达标的数量
            latex_fail_count = sum(1 for r in results if not r.get('latex_compile_success', False))
            similarity_fail_count = sum(1 for r in results 
                                      if r.get('latex_compile_success', False) and not r.get('similarity_above_threshold', False))
            
            # 计算相似度统计（仅编译成功的样本）
            compiled_results = [r for r in results if r.get('latex_compile_success', False)]
            if compiled_results:
                similarity_scores = [r['similarity_score'] for r in compiled_results]
                avg_similarity = np.mean(similarity_scores)
                median_similarity = np.median(similarity_scores)
                max_similarity = np.max(similarity_scores)
                min_similarity = np.min(similarity_scores)
            else:
                avg_similarity = median_similarity = max_similarity = min_similarity = 0.0
            
            # 生成报告内容
            report_content = f"""LaTeX公式相似度评估报告
{"="*60}
评估时间: {self._get_current_time()}
相似度阈值: {evaluation_summary['similarity_threshold']}

最终得分统计:
{"="*30}
总样本数: {evaluation_summary['total_samples']}
通过样本数: {evaluation_summary['passed_samples']}
失败样本数: {evaluation_summary['failed_samples']}
最终得分: {evaluation_summary['final_score']:.2f}%

失败原因分析:
{"="*30}
LaTeX编译失败: {latex_fail_count} 个
相似度不达标: {similarity_fail_count} 个

相似度统计 (仅编译成功的样本):
{"="*40}
编译成功样本数: {len(compiled_results)}
平均相似度: {avg_similarity:.4f}
中位数相似度: {median_similarity:.4f}
最高相似度: {max_similarity:.4f}
最低相似度: {min_similarity:.4f}

详细结果:
{"="*60}
"""
            
            # 按得分和相似度排序
            results.sort(key=lambda x: (x['score'], x.get('similarity_score', 0)), reverse=True)
            
            for i, result in enumerate(results, 1):
                txt_name = os.path.basename(result['txt_path'])
                score_status = "✓ 通过" if result['score'] == 1 else "✗ 未通过"
                
                report_content += f"{i:3d}. {score_status} - {txt_name}\n"
                
                if result.get('latex_compile_success', False):
                    similarity = result['similarity_score']
                    threshold_status = "达标" if result.get('similarity_above_threshold', False) else "不达标"
                    report_content += f"     相似度: {similarity:.4f} ({threshold_status})\n"
                    
                    if 'detailed_scores' in result:
                        report_content += f"     详细分数:\n"
                        for method, score in result['detailed_scores'].items():
                            if method != 'comprehensive':
                                report_content += f"       - {method}: {score:.4f}\n"
                else:
                    report_content += f"     LaTeX编译失败\n"
                
                if result.get('error'):
                    report_content += f"     错误: {result['error']}\n"
                
                if result.get('latex_content'):
                    preview = result['latex_content'][:80]
                    if len(result['latex_content']) > 80:
                        preview += "..."
                    report_content += f"     公式: {preview}\n"
                
                report_content += "\n"
            
            # 写入报告文件
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"\n详细评估报告已生成: {report_path}")
            
        except Exception as e:
            print(f"生成评估报告失败: {e}")
    
    def _get_current_time(self):
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 测试代码
if __name__ == "__main__":
    # 创建评估器，设置相似度阈值
    evaluator = LatexSimilarityEvaluator(dpi=300, fontsize=12, similarity_threshold=0.6)
    
    # 示例用法1: 单个文件评估
    print("="*60)
    print("单个文件评估示例")
    print("="*60)
    
    txt_path = "./samples/sample001.txt"
    ref_image_path = "./reference/sample001.png"
    
    if os.path.exists(txt_path) and os.path.exists(ref_image_path):
        result = evaluator.evaluate_single(txt_path, ref_image_path)
        
        print(f"评估结果:")
        print(f"最终得分: {result['score']} (0=未通过, 1=通过)")
        print(f"相似度分数: {result['similarity_score']:.4f}")
        print(f"阈值: {result['threshold']}")
        print(f"LaTeX编译成功: {result['latex_compile_success']}")
        print(f"相似度达标: {result['similarity_above_threshold']}")
        if result['error']:
            print(f"错误: {result['error']}")
    else:
        print("示例文件不存在，跳过单个文件测试")
    
    # 示例用法2: 批量评估
    print("\n" + "="*60)
    print("批量评估示例")
    print("="*60)
    
    txt_dir = "./samples_test2"
    ref_dir = "./output"
    report_path = "./evaluation_report.txt"
    
    if os.path.exists(txt_dir):
        # 可以动态调整阈值
        evaluator.set_threshold(0.99)  # 设置更严格的阈值
        
        summary = evaluator.evaluate_batch(
            txt_dir=txt_dir,
            reference_dir=ref_dir,
            output_report=report_path,
            cleanup=True
        )
        
        print(f"\n🎯 最终得分: {summary['final_score']:.2f}%")
        print(f"📊 通过率: {summary['passed_samples']}/{summary['total_samples']}")
    else:
        print("示例目录不存在，跳过批量测试")
        print("\n使用说明:")
        print("1. 准备txt文件目录，包含LaTeX公式")
        print("2. 准备参考图片目录，图片名称与txt文件对应")
        print("3. 设置相似度阈值: evaluator.set_threshold(0.6)")
        print("4. 调用 evaluator.evaluate_batch() 获得最终得分百分比")