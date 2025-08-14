import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional
from random import sample
from PIL import Image, ImageDraw, ImageFont
import subprocess
import tempfile
import glob
import re

@dataclass
class ConversionTask:
    """转换任务数据类"""
    input_file: str
    output_file: str
    latex_content: str
    task_id: int

@dataclass
class ConversionResult:
    """转换结果数据类"""
    task: ConversionTask
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0

class LatexConverter:
    """LaTeX转换器核心类"""
    
    def __init__(self, dpi=300, fontsize=14):
        self.dpi = dpi
        self.fontsize = fontsize
        self.latex_available = self._check_latex_availability()
        
        if self.latex_available:
            print("✓ 使用直接LaTeX渲染")
        else:
            print("⚠ LaTeX不可用，将生成错误提示图片")
    
    def _check_latex_availability(self):
        """检查LaTeX是否可用"""
        try:
            result = subprocess.run(['latex', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                result = subprocess.run(['dvipng', '--version'], 
                                      capture_output=True, text=True, timeout=10)
                return result.returncode == 0
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def convert_single(self, task: ConversionTask) -> ConversionResult:
        """转换单个任务"""
        start_time = time.time()
        
        try:
            os.makedirs(os.path.dirname(task.output_file), exist_ok=True)
            
            if not self.latex_available:
                self._create_blank_image(task.output_file)
                return ConversionResult(
                    task=task,
                    success=False,
                    error_message="LaTeX not available",
                    processing_time=time.time() - start_time
                )
            
            success = self._latex_to_image(task.latex_content, task.output_file)
            
            return ConversionResult(
                task=task,
                success=success,
                error_message=None if success else "LaTeX rendering failed",
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ConversionResult(
                task=task,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _latex_to_image(self, latex_string, output_path):
        """LaTeX转图片核心方法"""
        try:
            formula = self._prepare_latex_string(latex_string)
            
            latex_doc = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{amsfonts}}
\\usepackage[active,tightpage]{{preview}}
\\begin{{document}}
\\begin{{preview}}
${formula}$
\\end{{preview}}
\\end{{document}}
"""
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tex_file = os.path.join(tmpdir, 'formula.tex')
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(latex_doc)
                
                # 编译LaTeX
                result = subprocess.run(['latex', '-interaction=nonstopmode', 
                                       '-output-directory', tmpdir, tex_file],
                                      capture_output=True, text=True, cwd=tmpdir)
                
                if result.returncode == 0:
                    dvi_file = os.path.join(tmpdir, 'formula.dvi')
                    if os.path.exists(dvi_file):
                        result = subprocess.run(['dvipng', '-T', 'tight', '-D', str(self.dpi),
                                              '-bg', 'Transparent', '-o', output_path, dvi_file],
                                             capture_output=True, text=True)
                        return os.path.exists(output_path) and result.returncode == 0
                    
                return False
                    
        except Exception as e:
            print(f"LaTeX渲染失败: {e}")
            return False
    
    def _prepare_latex_string(self, latex_string):
        """准备LaTeX字符串格式"""
        formula = latex_string.strip()
        
        if formula.startswith('$$') and formula.endswith('$$'):
            formula = formula[2:-2].strip()
        elif formula.startswith('$') and formula.endswith('$'):
            formula = formula[1:-1].strip()
        
        lines = formula.split('\\\\')
        cleaned_lines = []
        for line in lines:
            cleaned_line = re.sub(r'\s+', ' ', line.strip())
            cleaned_lines.append(cleaned_line)
        formula = '\\\\'.join(cleaned_lines)
        
        return formula
    
    def _create_blank_image(self, output_path, background_color='white', size=(800, 400)):
        """创建空白图片"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            img = Image.new('RGB', size, background_color)
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.load_default()
                text = "LaTeX Parse Error"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (size[0] - text_width) // 2
                y = (size[1] - text_height) // 2
                draw.text((x, y), text, fill='red', font=font)
            except:
                pass
            
            img.save(output_path)
            
        except Exception as e:
            print(f"创建空白图片失败: {e}")

class TaskProducer:
    """任务生产者"""
    
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
    
    def produce_tasks(self) -> List[ConversionTask]:
        """生产所有转换任务"""
        pattern = os.path.join(self.input_folder, "sample*.txt")
        txt_files = glob.glob(pattern)
        
        if not txt_files:
            print(f"在 {self.input_folder} 中没有找到 sample*.txt 文件")
            return []
        
        txt_files.sort()
        print(f"📝 发现 {len(txt_files)} 个待处理文件")
        
        tasks = []
        for task_id, txt_file in enumerate(txt_files):
            try:
                base_name = os.path.splitext(os.path.basename(txt_file))[0]
                output_path = os.path.join(self.output_folder, f"{base_name}.png")
                
                with open(txt_file, 'r', encoding='utf-8') as f:
                    latex_content = f.read().strip()
                
                if latex_content:
                    task = ConversionTask(
                        input_file=txt_file,
                        output_file=output_path,
                        latex_content=latex_content,
                        task_id=task_id
                    )
                    tasks.append(task)
                else:
                    print(f"⚠ 跳过空文件: {txt_file}")
                    
            except Exception as e:
                print(f"✗ 读取文件失败 {txt_file}: {e}")
        
        return tasks

class ProgressMonitor:
    """进度监控器"""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def update(self, result: ConversionResult):
        """更新进度"""
        with self.lock:
            self.completed_tasks += 1
            if result.success:
                self.successful_tasks += 1
            else:
                self.failed_tasks += 1
            
            # 打印进度
            progress = (self.completed_tasks / self.total_tasks) * 100
            elapsed_time = time.time() - self.start_time
            
            if self.completed_tasks > 0:
                avg_time = elapsed_time / self.completed_tasks
                eta = avg_time * (self.total_tasks - self.completed_tasks)
                
                status = "✓" if result.success else "✗"
                filename = os.path.basename(result.task.input_file)
                
                print(f"{status} [{self.completed_tasks:3d}/{self.total_tasks}] "
                      f"({progress:5.1f}%) {filename} "
                      f"(用时: {result.processing_time:.2f}s, 预计剩余: {eta:.1f}s)")
    
    def get_summary(self):
        """获取汇总信息"""
        total_time = time.time() - self.start_time
        return {
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'total_time': total_time,
            'avg_time_per_task': total_time / max(self.completed_tasks, 1),
            'success_rate': (self.successful_tasks / max(self.total_tasks, 1)) * 100
        }

class LatexBatchProcessor:
    """LaTeX批处理器 - 生产者消费者模式"""
    
    def __init__(self, max_workers: int = None, dpi: int = 300, fontsize: int = 14):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.converter = LatexConverter(dpi, fontsize)
        self.results = []
        
        print(f"🚀 初始化批处理器，使用 {self.max_workers} 个工作线程")
    
    def process_folder(self, input_folder: str, output_folder: str) -> List[ConversionResult]:
        """处理文件夹中的所有LaTeX文件"""
        print(f"\n📁 输入文件夹: {input_folder}")
        print(f"📁 输出文件夹: {output_folder}")
        
        # 1. 生产者：生成所有任务
        producer = TaskProducer(input_folder, output_folder)
        tasks = producer.produce_tasks()
        
        if not tasks:
            print("❌ 没有找到可处理的文件")
            return []
        
        # 2. 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 3. 初始化进度监控
        monitor = ProgressMonitor(len(tasks))
        
        print(f"\n🔥 开始并行处理 {len(tasks)} 个任务...")
        print("-" * 80)
        
        # 4. 消费者：并行处理任务
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.converter.convert_single, task): task 
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    monitor.update(result)
                except Exception as e:
                    task = future_to_task[future]
                    error_result = ConversionResult(
                        task=task,
                        success=False,
                        error_message=f"处理异常: {str(e)}"
                    )
                    results.append(error_result)
                    monitor.update(error_result)
        
        print("-" * 80)
        
        # 5. 显示处理汇总
        summary = monitor.get_summary()
        self._print_summary(summary)
        
        self.results = results
        return results
    
    def _print_summary(self, summary):
        """打印处理汇总"""
        print(f"\n📊 处理完成汇总:")
        print(f"   总任务数: {summary['total_tasks']}")
        print(f"   成功转换: {summary['successful_tasks']}")
        print(f"   失败转换: {summary['failed_tasks']}")
        print(f"   成功率: {summary['success_rate']:.1f}%")
        print(f"   总用时: {summary['total_time']:.2f} 秒")
        print(f"   平均用时: {summary['avg_time_per_task']:.2f} 秒/任务")
        print(f"   处理速度: {summary['total_tasks']/summary['total_time']:.1f} 任务/秒")
    
    def generate_report(self, report_path: str = None) -> str:
        """生成详细报告"""
        if not self.results:
            return "没有处理结果可生成报告"
        
        if report_path is None:
            report_path = "./latex_conversion_report.txt"
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        total_time = sum(r.processing_time for r in self.results)
        avg_time = total_time / len(self.results)
        
        report_content = f"""LaTeX批量转换报告
{"="*60}
处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
工作线程数: {self.max_workers}

统计信息:
- 总文件数: {len(self.results)}
- 成功转换: {len(successful)}
- 失败转换: {len(failed)}
- 成功率: {len(successful)/len(self.results)*100:.1f}%
- 总处理时间: {total_time:.2f} 秒
- 平均处理时间: {avg_time:.2f} 秒/文件
- 处理速度: {len(self.results)/total_time:.1f} 文件/秒

成功转换的文件:
{"="*60}
"""
        
        for i, result in enumerate(successful, 1):
            filename = os.path.basename(result.task.input_file)
            report_content += f"{i:2d}. ✓ {filename} ({result.processing_time:.2f}s)\n"
        
        if failed:
            report_content += f"\n失败转换的文件:\n{'='*60}\n"
            for i, result in enumerate(failed, 1):
                filename = os.path.basename(result.task.input_file)
                report_content += f"{i:2d}. ✗ {filename} ({result.processing_time:.2f}s)\n"
                report_content += f"    错误: {result.error_message}\n"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"\n📄 详细报告已生成: {report_path}")
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
        
        return report_content

# 原有的辅助函数保持不变
def create_sample_files(sample_folder="./samples"):
    """创建示例sample*.txt文件"""
    os.makedirs(sample_folder, exist_ok=True)
    
    samples = [
        ("sample001.txt", r"E = mc^2"),
        ("sample002.txt", r"\sum_{i=1}^{n} x_i = \mu"),
        ("sample003.txt", r"\int_{0}^{1} x^2 dx = \frac{1}{3}"),
        ("sample004.txt", r"""
        \begin{bmatrix}
        a & b \\
        c & d
        \end{bmatrix}
        """),
        ("sample005.txt", r"""
        \begin{bmatrix}
        v_\pi(s_1) \\
        v_\pi(s_2) \\
        v_\pi(s_3) \\
        v_\pi(s_4)
        \end{bmatrix}
        """),
        ("sample006.txt", r"""
        \begin{bmatrix}
        v_\pi(s_1) \\
        v_\pi(s_2) \\
        v_\pi(s_3) \\
        v_\pi(s_4)
        \end{bmatrix} = 
        \begin{bmatrix}
        r_\pi(s_1) \\
        r_\pi(s_2) \\
        r_\pi(s_3) \\
        r_\pi(s_4)
        \end{bmatrix} + \gamma
        \begin{bmatrix}
        p_{11} & p_{12} & p_{13} & p_{14} \\
        p_{21} & p_{22} & p_{23} & p_{24} \\
        p_{31} & p_{32} & p_{33} & p_{34} \\
        p_{41} & p_{42} & p_{43} & p_{44}
        \end{bmatrix}
        \begin{bmatrix}
        v_\pi(s_1) \\
        v_\pi(s_2) \\
        v_\pi(s_3) \\
        v_\pi(s_4)
        \end{bmatrix}
        """),
        ("sample007.txt", r"""
        \underbrace{
        \begin{bmatrix}
        v_\pi(s_1) \\
        v_\pi(s_2) \\
        v_\pi(s_3) \\
        v_\pi(s_4)
        \end{bmatrix}
        }_{v_\pi} = 
        \underbrace{
        \begin{bmatrix}
        r_\pi(s_1) \\
        r_\pi(s_2) \\
        r_\pi(s_3) \\
        r_\pi(s_4)
        \end{bmatrix}
        }_{r_\pi} + \gamma
        \underbrace{
        \begin{bmatrix}
        p_\pi(s_1|s_1) & p_\pi(s_2|s_1) & p_\pi(s_3|s_1) & p_\pi(s_4|s_1) \\
        p_\pi(s_1|s_2) & p_\pi(s_2|s_2) & p_\pi(s_3|s_2) & p_\pi(s_4|s_2) \\
        p_\pi(s_1|s_3) & p_\pi(s_2|s_3) & p_\pi(s_3|s_3) & p_\pi(s_4|s_3) \\
        p_\pi(s_1|s_4) & p_\pi(s_2|s_4) & p_\pi(s_3|s_4) & p_\pi(s_4|s_4)
        \end{bmatrix}
        }_{P_\pi}
        \underbrace{
        \begin{bmatrix}
        v_\pi(s_1) \\
        v_\pi(s_2) \\
        v_\pi(s_3) \\
        v_\pi(s_4)
        \end{bmatrix}
        }_{v_\pi}
        """),
        ("sample008.txt", r"""
        Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
        """),
        ("sample009.txt", r"""
        P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
        """),
        ("sample010.txt", r"""
        \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
        """)
    ]
    
    print(f"📝 在 {sample_folder} 创建示例文件:")
    
    for filename, content in samples:
        file_path = os.path.join(sample_folder, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"   ✓ 创建: {filename}")
    
    print(f"📝 共创建了 {len(samples)} 个示例文件")
    return sample_folder

def check_latex_installation():
    """检查LaTeX安装状态"""
    print("🔍 检查LaTeX安装状态...")
    
    commands_to_check = [
        ('latex', ['latex', '--version']),
        ('dvipng', ['dvipng', '--version']),
        ('gs', ['gs', '--version'])
    ]
    
    all_ok = True
    for name, cmd in commands_to_check:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"   ✓ {name} 已安装")
            else:
                print(f"   ✗ {name} 未正确安装")
                all_ok = False
        except FileNotFoundError:
            print(f"   ✗ {name} 未找到")
            all_ok = False
        except Exception as e:
            print(f"   ✗ {name} 检查失败: {e}")
            all_ok = False
    
    return all_ok

# 主程序
if __name__ == "__main__":
    print("🚀 LaTeX批量转换工具 - 生产者消费者模式")
    print("=" * 60)
    
    # 检查LaTeX安装
    latex_ok = check_latex_installation()
    
    # 创建示例文件（可选）
    print(f"\n📝 准备示例文件...")
    sample_folder = "latex_formulas"
    if not os.path.exists(sample_folder):
        create_sample_files(sample_folder)
    else:
        pattern = os.path.join(sample_folder, "sample*.txt")
        existing_files = glob.glob(pattern)
        print(f"   发现 {len(existing_files)} 个现有sample文件")
    
    # 创建批处理器
    # 可以指定线程数，默认根据CPU核心数自动确定
    processor = LatexBatchProcessor(
        max_workers=64,  # 可以根据系统性能调整
        dpi=300,
        fontsize=12
    )
    
    # 开始批量处理
    print(f"\n🔥 开始批量处理...")
    output_folder = "latex_formulas/output"
    
    start_time = time.time()
    results = processor.process_folder(sample_folder, output_folder)
    total_time = time.time() - start_time
    
    # 生成详细报告
    if results:
        report_content = processor.generate_report()
        
        # 最终汇总
        successful = len([r for r in results if r.success])
        print(f"\n🎉 批量处理完成!")
        print(f"📈 性能统计:")
        print(f"   总文件数: {len(results)}")
        print(f"   成功转换: {successful}")
        print(f"   总耗时: {total_time:.2f} 秒")
        print(f"   处理速度: {len(results)/total_time:.1f} 文件/秒")
        print(f"   平均耗时: {total_time/len(results):.2f} 秒/文件")
        
        if not latex_ok:
            print(f"\n⚠ 注意: LaTeX未正确安装，生成的是错误提示图片")
    else:
        print("❌ 没有文件被处理")