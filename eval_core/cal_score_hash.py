import os
import sys
import hashlib
from pathlib import Path

class HashTestInterface:
    """
    哈希值测试接口
    专门用于比较LaTeX生成的图片与参考图片的哈希值
    """
    
    def __init__(self):
        self.supported_algorithms = ['md5', 'sha1', 'sha256']
    
    def calculate_file_hash(self, file_path, algorithm='md5'):
        """
        计算单个文件的哈希值
        
        Args:
            file_path (str): 文件路径
            algorithm (str): 哈希算法
            
        Returns:
            str: 哈希值，失败返回None
        """
        if not os.path.exists(file_path):
            return None
        
        try:
            if algorithm == 'md5':
                hasher = hashlib.md5()
            elif algorithm == 'sha1':
                hasher = hashlib.sha1()
            elif algorithm == 'sha256':
                hasher = hashlib.sha256()
            else:
                print(f"不支持的算法: {algorithm}")
                return None
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            print(f"计算哈希失败 {file_path}: {e}")
            return None
    
    def find_matching_files(self, txt_dir, ref_dir):
        """
        找到txt文件对应的参考图片文件
        
        Args:
            txt_dir (str): txt文件目录
            ref_dir (str): 参考图片目录
            
        Returns:
            list: 匹配的文件对列表
        """
        matches = []
        
        if not os.path.exists(txt_dir):
            print(f"txt目录不存在: {txt_dir}")
            return matches
        
        if not os.path.exists(ref_dir):
            print(f"参考目录不存在: {ref_dir}")
            return matches
        
        # 查找所有txt文件
        txt_files = []
        for file in os.listdir(txt_dir):
            if file.endswith('.txt'):
                txt_files.append(file)
        
        txt_files.sort()
        
        # 查找对应的图片文件
        for txt_file in txt_files:
            txt_path = os.path.join(txt_dir, txt_file)
            base_name = os.path.splitext(txt_file)[0]  # 去掉.txt扩展名
            
            # 尝试常见的图片扩展名
            ref_image = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                potential_ref = os.path.join(ref_dir, base_name + ext)
                if os.path.exists(potential_ref):
                    ref_image = potential_ref
                    break
            
            matches.append({
                'txt_file': txt_path,
                'ref_image': ref_image,
                'base_name': base_name
            })
        
        return matches
    
    def generate_images_from_txt(self, txt_dir, temp_output_dir):
        """
        从txt文件生成临时图片用于哈希比较
        
        Args:
            txt_dir (str): txt文件目录
            temp_output_dir (str): 临时输出目录
            
        Returns:
            list: 生成结果列表
        """
        # 导入LaTeX转换器
        try:
            from infer_core.latex2img_file import LatexToImage
        except ImportError:
            print("错误: 无法导入 latex2img_file 模块")
            print("请确保 latex2img_file.py 在当前目录或Python路径中")
            return []
        
        # 创建临时输出目录
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # 初始化LaTeX转换器
        converter = LatexToImage(dpi=300, fontsize=12)
        
        results = []
        
        # 查找所有txt文件
        txt_files = []
        for file in os.listdir(txt_dir):
            if file.endswith('.txt'):
                txt_files.append(file)
        
        txt_files.sort()
        
        print(f"开始从 {len(txt_files)} 个txt文件生成图片...")
        
        for i, txt_file in enumerate(txt_files, 1):
            txt_path = os.path.join(txt_dir, txt_file)
            base_name = os.path.splitext(txt_file)[0]
            output_path = os.path.join(temp_output_dir, f"{base_name}.png")
            
            print(f"[{i}/{len(txt_files)}] 处理: {txt_file}")
            
            try:
                # 读取LaTeX内容
                with open(txt_path, 'r', encoding='utf-8') as f:
                    latex_content = f.read().strip()
                
                if not latex_content:
                    print(f"  警告: txt文件为空")
                    results.append({
                        'txt_file': txt_path,
                        'generated_image': None,
                        'success': False,
                        'error': 'txt文件为空'
                    })
                    continue
                
                # 生成图片
                success = converter.latex_to_image(latex_content, output_path)
                
                if success and os.path.exists(output_path):
                    print(f"  ✓ 生成成功: {base_name}.png")
                    results.append({
                        'txt_file': txt_path,
                        'generated_image': output_path,
                        'success': True,
                        'latex_content': latex_content
                    })
                else:
                    print(f"  ✗ 生成失败")
                    results.append({
                        'txt_file': txt_path,
                        'generated_image': None,
                        'success': False,
                        'error': 'LaTeX编译失败'
                    })
                    
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                results.append({
                    'txt_file': txt_path,
                    'generated_image': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def compare_hash_values(self, txt_dir, ref_dir, report_path, algorithm='md5'):
        """
        比较txt生成的图片与参考图片的哈希值
        
        Args:
            txt_dir (str): txt文件目录
            ref_dir (str): 参考图片目录  
            report_path (str): 报告输出路径
            algorithm (str): 哈希算法
            
        Returns:
            dict: 比较结果统计
        """
        print("=" * 60)
        print("哈希值比较测试")
        print("=" * 60)
        print(f"txt目录: {txt_dir}")
        print(f"参考目录: {ref_dir}")
        print(f"哈希算法: {algorithm}")
        print(f"报告路径: {report_path}")
        
        # 创建临时目录生成图片
        temp_dir = "./temp_generated_images"
        
        try:
            # 1. 从txt文件生成图片
            print(f"\n步骤1: 从txt文件生成图片到临时目录 {temp_dir}")
            generation_results = self.generate_images_from_txt(txt_dir, temp_dir)
            
            # 2. 找到匹配的文件对
            print(f"\n步骤2: 查找匹配的文件对")
            matches = self.find_matching_files(txt_dir, ref_dir)
            
            # 3. 进行哈希比较
            print(f"\n步骤3: 进行哈希值比较")
            comparison_results = []
            identical_count = 0
            different_count = 0
            missing_count = 0
            generation_failed_count = 0
            
            for match in matches:
                base_name = match['base_name']
                txt_file = match['txt_file']
                ref_image = match['ref_image']
                generated_image = os.path.join(temp_dir, f"{base_name}.png")
                
                result = {
                    'base_name': base_name,
                    'txt_file': txt_file,
                    'ref_image': ref_image,
                    'generated_image': generated_image,
                    'ref_exists': ref_image is not None and os.path.exists(ref_image),
                    'generated_exists': os.path.exists(generated_image),
                    'ref_hash': None,
                    'generated_hash': None,
                    'identical': False,
                    'status': None
                }
                
                print(f"\n处理: {base_name}")
                
                # 检查参考图片是否存在
                if not result['ref_exists']:
                    print(f"  ✗ 参考图片不存在")
                    result['status'] = 'missing_reference'
                    missing_count += 1
                elif not result['generated_exists']:
                    print(f"  ✗ 生成图片失败")
                    result['status'] = 'generation_failed'
                    generation_failed_count += 1
                else:
                    # 计算哈希值
                    ref_hash = self.calculate_file_hash(ref_image, algorithm)
                    gen_hash = self.calculate_file_hash(generated_image, algorithm)
                    
                    result['ref_hash'] = ref_hash
                    result['generated_hash'] = gen_hash
                    
                    if ref_hash and gen_hash:
                        result['identical'] = (ref_hash == gen_hash)
                        
                        if result['identical']:
                            print(f"  ✓ 哈希值相同: {ref_hash}")
                            result['status'] = 'identical'
                            identical_count += 1
                        else:
                            print(f"  ✗ 哈希值不同:")
                            print(f"    参考: {ref_hash}")
                            print(f"    生成: {gen_hash}")
                            result['status'] = 'different'
                            different_count += 1
                    else:
                        print(f"  ✗ 无法计算哈希值")
                        result['status'] = 'hash_failed'
                        different_count += 1
                
                comparison_results.append(result)
            
            # 4. 生成统计信息
            total_count = len(comparison_results)
            summary = {
                'total_samples': total_count,
                'identical_count': identical_count,
                'different_count': different_count,
                'missing_reference_count': missing_count,
                'generation_failed_count': generation_failed_count,
                'identical_rate': (identical_count / total_count * 100) if total_count > 0 else 0,
                'comparison_results': comparison_results,
                'algorithm': algorithm
            }
            
            # 5. 生成报告
            self._generate_hash_report(summary, report_path)
            
            # 6. 打印总结
            print(f"\n" + "=" * 60)
            print("哈希比较完成")
            print("=" * 60)
            print(f"总样本数: {total_count}")
            print(f"哈希相同: {identical_count}")
            print(f"哈希不同: {different_count}")
            print(f"缺少参考: {missing_count}")
            print(f"生成失败: {generation_failed_count}")
            print(f"相同率: {summary['identical_rate']:.2f}%")
            
            return summary
            
        finally:
            # 清理临时目录
            try:
                import shutil
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"\n临时目录已清理: {temp_dir}")
            except Exception as e:
                print(f"清理临时目录失败: {e}")
    
    def _generate_hash_report(self, summary, report_path):
        """
        生成哈希比较报告
        
        Args:
            summary (dict): 比较结果统计
            report_path (str): 报告文件路径
        """
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            report_content = f"""哈希值比较测试报告
{"=" * 60}
测试时间: {self._get_current_time()}
哈希算法: {summary['algorithm']}

总体统计:
{"=" * 30}
总样本数: {summary['total_samples']}
哈希相同: {summary['identical_count']}
哈希不同: {summary['different_count']}
缺少参考图片: {summary['missing_reference_count']}
生成失败: {summary['generation_failed_count']}
相同率: {summary['identical_rate']:.2f}%

详细结果:
{"=" * 60}
"""
            
            # 按状态分组显示
            status_groups = {
                'identical': [],
                'different': [],
                'missing_reference': [],
                'generation_failed': [],
                'hash_failed': []
            }
            
            for result in summary['comparison_results']:
                status = result.get('status', 'unknown')
                if status in status_groups:
                    status_groups[status].append(result)
            
            # 显示相同的文件
            if status_groups['identical']:
                report_content += f"\n✓ 哈希值相同的文件 ({len(status_groups['identical'])} 个):\n"
                for result in status_groups['identical']:
                    report_content += f"  {result['base_name']}: {result['ref_hash']}\n"
            
            # 显示不同的文件
            if status_groups['different']:
                report_content += f"\n✗ 哈希值不同的文件 ({len(status_groups['different'])} 个):\n"
                for result in status_groups['different']:
                    report_content += f"  {result['base_name']}:\n"
                    report_content += f"    参考: {result['ref_hash']}\n"
                    report_content += f"    生成: {result['generated_hash']}\n"
            
            # 显示缺少参考的文件
            if status_groups['missing_reference']:
                report_content += f"\n⚠ 缺少参考图片的文件 ({len(status_groups['missing_reference'])} 个):\n"
                for result in status_groups['missing_reference']:
                    report_content += f"  {result['base_name']}\n"
            
            # 显示生成失败的文件
            if status_groups['generation_failed']:
                report_content += f"\n❌ 生成失败的文件 ({len(status_groups['generation_failed'])} 个):\n"
                for result in status_groups['generation_failed']:
                    report_content += f"  {result['base_name']}\n"
            
            # 写入报告文件
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"\n哈希比较报告已生成: {report_path}")
            
        except Exception as e:
            print(f"生成报告失败: {e}")
    
    def _get_current_time(self):
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函数"""
    # 配置路径
    txt_dir = "./samples_test2"
    ref_dir = "./output"  
    report_path = "./hash_comparison_report.txt"
    
    print("LaTeX图片哈希值比较测试")
    print("=" * 60)
    print(f"配置信息:")
    print(f"  txt文件目录: {txt_dir}")
    print(f"  参考图片目录: {ref_dir}")
    print(f"  报告输出路径: {report_path}")
    
    # 检查目录是否存在
    if not os.path.exists(txt_dir):
        print(f"\n错误: txt目录不存在 - {txt_dir}")
        return
    
    if not os.path.exists(ref_dir):
        print(f"\n错误: 参考目录不存在 - {ref_dir}")
        return
    
    # 创建测试接口
    hash_tester = HashTestInterface()
    
    # 选择哈希算法
    algorithm = 'md5'  # 可以改为 'sha1' 或 'sha256'
    
    # 执行哈希比较测试
    try:
        summary = hash_tester.compare_hash_values(txt_dir, ref_dir, report_path, algorithm)
        
        # 显示结论
        print(f"\n" + "=" * 60)
        print("测试结论")
        print("=" * 60)
        
        if summary['identical_rate'] == 100:
            print("🎉 所有文件的哈希值都相同！")
            print("   这说明生成的图片与参考图片完全一致。")
        elif summary['identical_rate'] > 90:
            print("✅ 大部分文件的哈希值相同。")
            print(f"   相同率: {summary['identical_rate']:.2f}%")
        elif summary['identical_rate'] > 50:
            print("⚠️  部分文件的哈希值相同。")
            print(f"   相同率: {summary['identical_rate']:.2f}%")
        else:
            print("❌ 大部分文件的哈希值不同。")
            print(f"   相同率: {summary['identical_rate']:.2f}%")
        
        if summary['different_count'] > 0:
            print(f"\n💡 发现 {summary['different_count']} 个文件哈希值不同")
            print("   可能原因:")
            print("   1. LaTeX编译过程中的微小差异")
            print("   2. 图片生成时间戳或元数据不同")
            print("   3. 浮点数精度导致的像素微小差异")
        
    except Exception as e:
        print(f"\n执行测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()