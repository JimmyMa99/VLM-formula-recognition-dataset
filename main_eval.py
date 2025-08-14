import os
import sys
import shutil
from pathlib import Path

# 导入评估模块
sys.path.append('./eval_core')
sys.path.append('./eval_core')
from eval_core.cal_score_hash import HashTestInterface
from eval_core.cal_score import LatexSimilarityEvaluator

class ComprehensiveEvaluator:
    """
    综合评估器
    结合哈希值比较和图像相似度计算，通过加权方式得出最终得分
    """
    
    def __init__(self, hash_weight=0.5, similarity_weight=0.5, similarity_threshold=0.6):
        """
        初始化综合评估器
        
        Args:
            hash_weight (float): 哈希比较的权重
            similarity_weight (float): 相似度计算的权重
            similarity_threshold (float): 相似度阈值
        """
        self.hash_weight = hash_weight
        self.similarity_weight = similarity_weight
        self.similarity_threshold = similarity_threshold
        
        # 确保权重和为1
        total_weight = hash_weight + similarity_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"警告: 权重和不为1 ({total_weight})，将自动归一化")
            self.hash_weight = hash_weight / total_weight
            self.similarity_weight = similarity_weight / total_weight
        
        print(f"综合评估器初始化完成")
        print(f"哈希比较权重: {self.hash_weight}")
        print(f"相似度计算权重: {self.similarity_weight}")
        print(f"相似度阈值: {self.similarity_threshold}")
        
        # 初始化子评估器
        self.hash_tester = HashTestInterface()
        self.similarity_evaluator = LatexSimilarityEvaluator(
            dpi=300, 
            fontsize=12, 
            similarity_threshold=similarity_threshold
        )
    
    def evaluate_comprehensive(self, txt_dir, ref_dir, output_report=None, keep_temp_images=False):
        """
        综合评估：结合哈希比较和相似度计算
        
        Args:
            txt_dir (str): txt文件目录
            ref_dir (str): 参考图片目录
            output_report (str): 输出报告路径
            keep_temp_images (bool): 是否保留临时生成的图片
            
        Returns:
            dict: 综合评估结果
        """
        print("=" * 80)
        print("综合评估开始")
        print("=" * 80)
        print(f"txt目录: {txt_dir}")
        print(f"参考目录: {ref_dir}")
        print(f"哈希权重: {self.hash_weight}")
        print(f"相似度权重: {self.similarity_weight}")
        
        # 检查目录
        if not os.path.exists(txt_dir):
            raise FileNotFoundError(f"txt目录不存在: {txt_dir}")
        if not os.path.exists(ref_dir):
            raise FileNotFoundError(f"参考目录不存在: {ref_dir}")
        
        # 首先统计总的测试集数量
        total_test_samples = self._count_total_samples(txt_dir)
        print(f"总测试样本数: {total_test_samples}")
        
        # 创建临时目录保存生成的图片
        temp_generated_dir = "./temp_comprehensive_eval"
        os.makedirs(temp_generated_dir, exist_ok=True)
        
        try:
            # 第一步：生成图片并进行哈希比较
            print(f"\n{'='*60}")
            print("第一步：哈希值比较评估")
            print(f"{'='*60}")
            
            hash_results = self._evaluate_hash_comparison(txt_dir, ref_dir, temp_generated_dir, total_test_samples)
            
            # 第二步：进行相似度计算 (基于总样本数，不是成功生成的数量)
            print(f"\n{'='*60}")
            print("第二步：图像相似度评估")
            print(f"{'='*60}")
            
            similarity_results = self._evaluate_similarity_correct(txt_dir, ref_dir, temp_generated_dir, total_test_samples)
            
            # 第三步：综合计算最终得分
            print(f"\n{'='*60}")
            print("第三步：综合得分计算")
            print(f"{'='*60}")
            
            comprehensive_results = self._calculate_comprehensive_score(hash_results, similarity_results)
            
            # 第四步：生成综合报告
            if output_report:
                self._generate_comprehensive_report(comprehensive_results, output_report)
            
            # 打印最终结果
            self._print_final_results(comprehensive_results)
            
            return comprehensive_results
            
        finally:
            # 清理临时文件
            if not keep_temp_images:
                try:
                    if os.path.exists(temp_generated_dir):
                        shutil.rmtree(temp_generated_dir)
                        print(f"\n临时目录已清理: {temp_generated_dir}")
                except Exception as e:
                    print(f"清理临时目录失败: {e}")
            else:
                print(f"\n生成的图片已保存到: {temp_generated_dir}")
    
    def _count_total_samples(self, txt_dir):
        """
        统计总的测试样本数量
        
        Args:
            txt_dir (str): txt文件目录
            
        Returns:
            int: 总样本数
        """
        txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
        return len(txt_files)
    
    def _evaluate_hash_comparison(self, txt_dir, ref_dir, temp_generated_dir, total_samples):
        """
        执行哈希值比较评估
        
        Args:
            txt_dir (str): txt文件目录
            ref_dir (str): 参考图片目录
            temp_generated_dir (str): 临时生成图片目录
            total_samples (int): 总样本数
            
        Returns:
            dict: 哈希比较结果
        """
        # 生成图片
        generation_results = self.hash_tester.generate_images_from_txt(txt_dir, temp_generated_dir)
        
        # 获取所有txt文件列表
        txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
        txt_files.sort()
        
        # 进行哈希比较，确保覆盖所有测试样本
        hash_comparison_results = []
        hash_identical_count = 0
        
        for txt_file in txt_files:
            base_name = os.path.splitext(txt_file)[0]
            
            # 查找对应的参考图片
            ref_image = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                potential_ref = os.path.join(ref_dir, base_name + ext)
                if os.path.exists(potential_ref):
                    ref_image = potential_ref
                    break
            
            generated_image = os.path.join(temp_generated_dir, f"{base_name}.png")
            
            result = {
                'base_name': base_name,
                'txt_file': os.path.join(txt_dir, txt_file),
                'ref_image': ref_image,
                'generated_image': generated_image,
                'ref_exists': ref_image is not None,
                'generated_exists': os.path.exists(generated_image),
                'hash_identical': False,
                'hash_score': 0,  # 0 或 1
                'error': None
            }
            
            if not result['ref_exists']:
                result['error'] = '参考图片不存在'
                print(f"  ✗ {base_name}: 参考图片不存在")
            elif not result['generated_exists']:
                result['error'] = 'LaTeX编译失败，无法生成图片'
                print(f"  ✗ {base_name}: LaTeX编译失败")
            else:
                # 计算哈希值比较
                ref_hash = self.hash_tester.calculate_file_hash(ref_image)
                gen_hash = self.hash_tester.calculate_file_hash(generated_image)
                
                if ref_hash and gen_hash and ref_hash == gen_hash:
                    result['hash_identical'] = True
                    result['hash_score'] = 1
                    hash_identical_count += 1
                    print(f"  ✓ {base_name}: 哈希值相同")
                else:
                    print(f"  ✗ {base_name}: 哈希值不同")
            
            hash_comparison_results.append(result)
        
        # 确保统计的是总样本数，不是成功的数量
        hash_success_rate = (hash_identical_count / total_samples * 100) if total_samples > 0 else 0
        
        print(f"\n哈希比较结果:")
        print(f"  总样本数: {total_samples}")
        print(f"  哈希相同: {hash_identical_count}")
        print(f"  成功率: {hash_success_rate:.2f}% (基于总样本数)")
        
        return {
            'total_samples': total_samples,
            'identical_count': hash_identical_count,
            'success_rate': hash_success_rate,
            'results': hash_comparison_results
        }
    
    def _evaluate_similarity_correct(self, txt_dir, ref_dir, temp_generated_dir, total_samples):
        """
        执行图像相似度评估 (修正版本：基于总样本数计算)
        
        Args:
            txt_dir (str): txt文件目录
            ref_dir (str): 参考图片目录
            temp_generated_dir (str): 临时生成图片目录
            total_samples (int): 总样本数
            
        Returns:
            dict: 相似度评估结果
        """
        # 获取所有txt文件列表
        txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
        txt_files.sort()
        
        similarity_results = []
        similarity_passed_count = 0
        
        for txt_file in txt_files:
            base_name = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(txt_dir, txt_file)
            
            # 查找对应的参考图片
            ref_image = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                potential_ref = os.path.join(ref_dir, base_name + ext)
                if os.path.exists(potential_ref):
                    ref_image = potential_ref
                    break
            
            generated_image = os.path.join(temp_generated_dir, f"{base_name}.png")
            
            result = {
                'base_name': base_name,
                'txt_file': txt_path,
                'ref_image': ref_image,
                'generated_image': generated_image,
                'ref_exists': ref_image is not None,
                'generated_exists': os.path.exists(generated_image),
                'similarity_score': 0.0,
                'similarity_passed': False,
                'similarity_binary_score': 0,  # 0 或 1
                'error': None
            }
            
            if not result['ref_exists']:
                result['error'] = '参考图片不存在'
                print(f"  ✗ {base_name}: 参考图片不存在")
            elif not result['generated_exists']:
                result['error'] = 'LaTeX编译失败，无法生成图片'
                print(f"  ✗ {base_name}: LaTeX编译失败，相似度得分=0")
                # 注意：这里similarity_score保持0.0，similarity_binary_score保持0
            else:
                try:
                    # 计算相似度
                    similarity_scores = self.similarity_evaluator.similarity_calculator.comprehensive_similarity(
                        generated_image, ref_image
                    )
                    
                    result['similarity_score'] = similarity_scores['comprehensive']
                    result['detailed_scores'] = similarity_scores
                    
                    # 根据阈值判断是否通过
                    if result['similarity_score'] >= self.similarity_threshold:
                        result['similarity_passed'] = True
                        result['similarity_binary_score'] = 1
                        similarity_passed_count += 1
                        print(f"  ✓ {base_name}: 相似度 {result['similarity_score']:.4f} (≥{self.similarity_threshold})")
                    else:
                        print(f"  ✗ {base_name}: 相似度 {result['similarity_score']:.4f} (<{self.similarity_threshold})")
                        
                except Exception as e:
                    result['error'] = f'相似度计算失败: {str(e)}'
                    print(f"  ✗ {base_name}: 相似度计算失败")
            
            similarity_results.append(result)
        
        # 关键修正：基于总样本数计算成功率，而不是成功生成的图片数
        similarity_success_rate = (similarity_passed_count / total_samples * 100) if total_samples > 0 else 0
        
        print(f"\n相似度比较结果:")
        print(f"  总样本数: {total_samples}")
        print(f"  相似度通过: {similarity_passed_count}")
        print(f"  成功率: {similarity_success_rate:.2f}% (基于总样本数)")
        
        return {
            'total_samples': total_samples,
            'passed_count': similarity_passed_count,
            'success_rate': similarity_success_rate,
            'results': similarity_results
        }
    
    def _calculate_comprehensive_score(self, hash_results, similarity_results):
        """
        计算综合得分
        
        Args:
            hash_results (dict): 哈希比较结果
            similarity_results (dict): 相似度比较结果
            
        Returns:
            dict: 综合评估结果
        """
        # 两个结果应该有相同的总样本数
        total_samples = hash_results['total_samples']
        assert total_samples == similarity_results['total_samples'], "哈希和相似度评估的样本数不一致"
        
        if total_samples == 0:
            return {
                'total_samples': 0,
                'final_score': 0.0,
                'hash_component': 0.0,
                'similarity_component': 0.0,
                'detailed_results': []
            }
        
        # 计算加权得分
        hash_component_score = hash_results['success_rate'] * self.hash_weight
        similarity_component_score = similarity_results['success_rate'] * self.similarity_weight
        final_score = hash_component_score + similarity_component_score
        
        # 合并详细结果
        detailed_results = []
        hash_dict = {r['base_name']: r for r in hash_results['results']}
        similarity_dict = {r['base_name']: r for r in similarity_results['results']}
        
        all_base_names = set(hash_dict.keys()) | set(similarity_dict.keys())
        
        for base_name in sorted(all_base_names):
            hash_result = hash_dict.get(base_name, {'hash_score': 0, 'hash_identical': False})
            similarity_result = similarity_dict.get(base_name, {'similarity_binary_score': 0, 'similarity_score': 0.0})
            
            # 计算该样本的综合得分
            sample_hash_score = hash_result['hash_score'] * self.hash_weight
            sample_similarity_score = similarity_result['similarity_binary_score'] * self.similarity_weight
            sample_final_score = sample_hash_score + sample_similarity_score
            
            detailed_results.append({
                'base_name': base_name,
                'hash_score': hash_result['hash_score'],
                'similarity_binary_score': similarity_result['similarity_binary_score'],
                'similarity_raw_score': similarity_result.get('similarity_score', 0.0),
                'weighted_hash_score': sample_hash_score,
                'weighted_similarity_score': sample_similarity_score,
                'final_score': sample_final_score,
                'hash_identical': hash_result.get('hash_identical', False),
                'similarity_passed': similarity_result.get('similarity_passed', False),
                'hash_error': hash_result.get('error'),
                'similarity_error': similarity_result.get('error')
            })
        
        return {
            'total_samples': total_samples,
            'final_score': final_score,
            'hash_component': hash_component_score,
            'similarity_component': similarity_component_score,
            'hash_success_rate': hash_results['success_rate'],
            'similarity_success_rate': similarity_results['success_rate'],
            'detailed_results': detailed_results,
            'weights': {
                'hash_weight': self.hash_weight,
                'similarity_weight': self.similarity_weight
            }
        }
    
    def _print_final_results(self, comprehensive_results):
        """
        打印最终结果
        
        Args:
            comprehensive_results (dict): 综合评估结果
        """
        print(f"\n{'='*80}")
        print("最终综合评估结果")
        print(f"{'='*80}")
        
        print(f"总样本数: {comprehensive_results['total_samples']}")
        print(f"哈希比较成功率: {comprehensive_results['hash_success_rate']:.2f}% (基于总样本数)")
        print(f"相似度比较成功率: {comprehensive_results['similarity_success_rate']:.2f}% (基于总样本数)")
        print(f"")
        print(f"加权得分组成:")
        print(f"  哈希组件得分: {comprehensive_results['hash_component']:.2f} (权重: {comprehensive_results['weights']['hash_weight']})")
        print(f"  相似度组件得分: {comprehensive_results['similarity_component']:.2f} (权重: {comprehensive_results['weights']['similarity_weight']})")
        print(f"")
        print(f"🎯 最终综合得分: {comprehensive_results['final_score']:.2f}")
        
        # 统计各种情况的样本数
        both_passed = sum(1 for r in comprehensive_results['detailed_results'] 
                         if r['hash_identical'] and r['similarity_passed'])
        only_hash_passed = sum(1 for r in comprehensive_results['detailed_results'] 
                              if r['hash_identical'] and not r['similarity_passed'])
        only_similarity_passed = sum(1 for r in comprehensive_results['detailed_results'] 
                                    if not r['hash_identical'] and r['similarity_passed'])
        both_failed = sum(1 for r in comprehensive_results['detailed_results'] 
                         if not r['hash_identical'] and not r['similarity_passed'])
        
        print(f"\n样本分布:")
        print(f"  两项都通过: {both_passed}")
        print(f"  仅哈希通过: {only_hash_passed}")
        print(f"  仅相似度通过: {only_similarity_passed}")
        print(f"  两项都失败: {both_failed}")
        
        # 统计失败原因
        latex_compile_failed = sum(1 for r in comprehensive_results['detailed_results'] 
                                  if r.get('hash_error') == 'LaTeX编译失败，无法生成图片' or 
                                     r.get('similarity_error') == 'LaTeX编译失败，无法生成图片')
        
        print(f"\n失败原因统计:")
        print(f"  LaTeX编译失败: {latex_compile_failed}")
        print(f"  其他原因失败: {both_failed - latex_compile_failed}")
    
    def _generate_comprehensive_report(self, comprehensive_results, report_path):
        """
        生成综合评估报告
        
        Args:
            comprehensive_results (dict): 综合评估结果
            report_path (str): 报告文件路径
        """
        try:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            report_content = f"""LaTeX图片综合相似度评估报告
{'='*80}
评估时间: {current_time}
评估方法: 哈希比较 + 图像相似度 (加权)

权重配置:
{'='*40}
哈希比较权重: {comprehensive_results['weights']['hash_weight']}
相似度计算权重: {comprehensive_results['weights']['similarity_weight']}
相似度阈值: {self.similarity_threshold}

总体统计:
{'='*40}
总样本数: {comprehensive_results['total_samples']}
哈希比较成功率: {comprehensive_results['hash_success_rate']:.2f}% (基于总样本数)
相似度比较成功率: {comprehensive_results['similarity_success_rate']:.2f}% (基于总样本数)

加权得分:
{'='*40}
哈希组件得分: {comprehensive_results['hash_component']:.2f}
相似度组件得分: {comprehensive_results['similarity_component']:.2f}
最终综合得分: {comprehensive_results['final_score']:.2f}

详细结果:
{'='*80}
"""
            
            # 按最终得分排序
            sorted_results = sorted(comprehensive_results['detailed_results'], 
                                  key=lambda x: x['final_score'], reverse=True)
            
            for i, result in enumerate(sorted_results, 1):
                status_hash = "✓" if result['hash_identical'] else "✗"
                status_sim = "✓" if result['similarity_passed'] else "✗"
                
                report_content += f"{i:3d}. {result['base_name']}\n"
                report_content += f"     哈希比较: {status_hash} ({result['hash_score']})\n"
                report_content += f"     相似度: {status_sim} ({result['similarity_raw_score']:.4f})\n"
                report_content += f"     加权得分: 哈希={result['weighted_hash_score']:.2f} + 相似度={result['weighted_similarity_score']:.2f} = {result['final_score']:.2f}\n"
                
                # 添加错误信息
                if result.get('hash_error'):
                    report_content += f"     哈希错误: {result['hash_error']}\n"
                if result.get('similarity_error'):
                    report_content += f"     相似度错误: {result['similarity_error']}\n"
                
                report_content += "\n"
            
            # 写入报告文件
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"\n📊 综合评估报告已生成: {report_path}")
            
        except Exception as e:
            print(f"生成报告失败: {e}")


def main():
    """主函数"""
    # 配置路径
    txt_dir = "./data/samples_test2"
    ref_dir = "./data/output_eval"
    report_path = "./comprehensive_evaluation_report.txt"
    
    # 配置权重
    hash_weight = 0.5
    similarity_weight = 0.5
    similarity_threshold = 0.99  # 相似度阈值
    
    print("LaTeX图片综合相似度评估")
    print("=" * 80)
    print(f"配置信息:")
    print(f"  txt文件目录: {txt_dir}")
    print(f"  参考图片目录: {ref_dir}")
    print(f"  报告输出路径: {report_path}")
    print(f"  哈希比较权重: {hash_weight}")
    print(f"  相似度计算权重: {similarity_weight}")
    print(f"  相似度阈值: {similarity_threshold}")
    print(f"  ⚠️  注意: 无法生成图片的样本计为0分，分母仍为总样本数")
    
    # 检查目录是否存在
    if not os.path.exists(txt_dir):
        print(f"\n❌ 错误: txt目录不存在 - {txt_dir}")
        return
    
    if not os.path.exists(ref_dir):
        print(f"\n❌ 错误: 参考目录不存在 - {ref_dir}")
        return
    
    try:
        # 创建综合评估器
        evaluator = ComprehensiveEvaluator(
            hash_weight=hash_weight,
            similarity_weight=similarity_weight,
            similarity_threshold=similarity_threshold
        )
        
        # 执行综合评估
        results = evaluator.evaluate_comprehensive(
            txt_dir=txt_dir,
            ref_dir=ref_dir,
            output_report=report_path,
            keep_temp_images=False  # 设为True可保留生成的临时图片
        )
        
        print(f"\n🎉 评估完成！最终得分: {results['final_score']:.2f}")
        
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()