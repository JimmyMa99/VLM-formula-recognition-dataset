import os
from random import sample
from PIL import Image, ImageDraw, ImageFont
import subprocess
import tempfile
import glob
import re


class LatexToImage:
    def __init__(self, dpi=300, fontsize=14):
        """
        初始化LaTeX转图片类（全部使用直接LaTeX渲染）
        """
        self.dpi = dpi
        self.fontsize = fontsize

        # 检查LaTeX是否可用
        self.latex_available = self._check_latex_availability()

        if self.latex_available:
            print("info: ✓ 使用直接LaTeX渲染")
        else:
            print("error:⚠ LaTeX不可用，将生成错误提示图片")
            exit(1)

    def _check_latex_availability(self):
        """检查LaTeX是否可用"""
        try:
            # 检查latex命令
            result = subprocess.run(
                ["latex", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # 检查dvipng
                result = subprocess.run(
                    ["dvipng", "--version"], capture_output=True, text=True, timeout=10
                )
                return result.returncode == 0
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def latex_to_image(self, latex_string, output_path, background_color="white"):
        """
        使用直接LaTeX渲染将公式转换为图片
        """
        if not self.latex_available:
            self._create_blank_image(output_path, background_color)
            return False

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            formula = self._prepare_latex_string(latex_string)
            print(f"处理后的公式: {formula[:100]}...")

            # 创建完整的LaTeX文档
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
                tex_file = os.path.join(tmpdir, "formula.tex")
                with open(tex_file, "w", encoding="utf-8") as f:
                    f.write(latex_doc)

                # 编译LaTeX
                result = subprocess.run(
                    [
                        "latex",
                        "-interaction=nonstopmode",
                        "-output-directory",
                        tmpdir,
                        tex_file,
                    ],
                    capture_output=True,
                    text=True,
                    cwd=tmpdir,
                )

                if result.returncode == 0:
                    dvi_file = os.path.join(tmpdir, "formula.dvi")
                    if os.path.exists(dvi_file):
                        # 转换为PNG，使用透明背景
                        result = subprocess.run(
                            [
                                "dvipng",
                                "-T",
                                "tight",
                                "-D",
                                str(self.dpi),
                                "-bg",
                                "Transparent",
                                "-o",
                                output_path,
                                dvi_file,
                            ],
                            capture_output=True,
                            text=True,
                        )
                        return os.path.exists(output_path) and result.returncode == 0
                    else:
                        print("DVI文件未生成")
                        return False
                else:
                    print(f"LaTeX编译失败: {result.stderr}")
                    return False

        except Exception as e:
            print(f"LaTeX渲染失败: {e}")
            self._create_blank_image(output_path, background_color)
            return False

    def _prepare_latex_string(self, latex_string):
        """准备LaTeX字符串格式"""
        formula = latex_string.strip()

        # 移除外层的$$或$
        if formula.startswith("$$") and formula.endswith("$$"):
            formula = formula[2:-2].strip()
        elif formula.startswith("$") and formula.endswith("$"):
            formula = formula[1:-1].strip()

        # 清理多余的空白，但保留必要的换行
        import re

        # 保留\\换行符，但清理其他多余空白
        lines = formula.split("\\\\")
        cleaned_lines = []
        for line in lines:
            cleaned_line = re.sub(r"\s+", " ", line.strip())
            cleaned_lines.append(cleaned_line)
        formula = "\\\\".join(cleaned_lines)

        return formula

    def _create_blank_image(
        self, output_path, background_color="white", size=(800, 400)
    ):
        """创建空白图片"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            img = Image.new("RGB", size, background_color)
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.load_default()
                text = "LaTeX Parse Error"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (size[0] - text_width) // 2
                y = (size[1] - text_height) // 2
                draw.text((x, y), text, fill="red", font=font)
            except:
                pass

            img.save(output_path)

        except Exception as e:
            print(f"创建空白图片失败: {e}")

    def process_folder(self, input_folder, output_folder):
        """
        读取文件夹中的sample*.txt文件并生成对应的图片
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 查找所有sample*.txt文件
        pattern = os.path.join(input_folder, "sample*.txt")
        txt_files = glob.glob(pattern)

        if not txt_files:
            print(f"在 {input_folder} 中没有找到 sample*.txt 文件")
            return []

        # 按文件名排序
        txt_files.sort()

        print(f"找到 {len(txt_files)} 个sample*.txt文件")

        results = []

        for txt_file in txt_files:
            # 获取文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            output_path = os.path.join(output_folder, f"{base_name}.png")

            try:
                # 读取LaTeX公式
                with open(txt_file, "r", encoding="utf-8") as f:
                    latex_content = f.read().strip()

                if not latex_content:
                    print(f"警告: {txt_file} 文件为空")
                    continue

                print(f"\n处理文件: {txt_file}")
                success = self.latex_to_image(latex_content, output_path)

                result_info = {
                    "input_file": txt_file,
                    "output_file": output_path,
                    "success": success,
                    "formula_preview": latex_content[:100]
                    + ("..." if len(latex_content) > 100 else ""),
                }
                results.append(result_info)

                if success:
                    print(f"✓ 成功生成: {output_path}")
                else:
                    print(f"✗ 生成失败: {base_name}")

            except Exception as e:
                print(f"✗ 处理文件 {txt_file} 时出错: {e}")
                result_info = {
                    "input_file": txt_file,
                    "output_file": output_path,
                    "success": False,
                    "error": str(e),
                }
                results.append(result_info)

        return results

    def generate_report(self, results, report_path=None):
        """生成处理报告"""
        if report_path is None:
            report_path = "./latex_conversion_report.txt"

        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)

        report_content = f"""LaTeX转图片处理报告
{"=" * 50}
总文件数: {total_count}
成功转换: {success_count}
失败转换: {total_count - success_count}
成功率: {success_count / total_count * 100:.1f}%

详细结果:
{"=" * 50}
"""

        for i, result in enumerate(results, 1):
            status = "✓ 成功" if result["success"] else "✗ 失败"
            report_content += (
                f"{i:2d}. {status} - {os.path.basename(result['input_file'])}\n"
            )
            report_content += f"    输入: {result['input_file']}\n"
            report_content += f"    输出: {result['output_file']}\n"

            if "formula_preview" in result:
                report_content += f"    公式: {result['formula_preview']}\n"

            if "error" in result:
                report_content += f"    错误: {result['error']}\n"

            report_content += "\n"

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"\n报告已生成: {report_path}")
        except Exception as e:
            print(f"生成报告失败: {e}")

        return report_content


def create_sample_files(sample_folder="./samples"):
    """创建示例sample*.txt文件"""
    os.makedirs(sample_folder, exist_ok=True)

    samples = [
        # sample001.txt - 基本公式
        ("sample001.txt", r"E = mc^2"),
        # sample002.txt - 求和公式
        ("sample002.txt", r"\sum_{i=1}^{n} x_i = \mu"),
        # sample003.txt - 积分公式
        ("sample003.txt", r"\int_{0}^{1} x^2 dx = \frac{1}{3}"),
        # sample004.txt - 简单矩阵
        (
            "sample004.txt",
            r"""
        \begin{bmatrix}
        a & b \\
        c & d
        \end{bmatrix}
        """,
        ),
        # sample005.txt - 状态值函数向量
        (
            "sample005.txt",
            r"""
        \begin{bmatrix}
        v_\pi(s_1) \\
        v_\pi(s_2) \\
        v_\pi(s_3) \\
        v_\pi(s_4)
        \end{bmatrix}
        """,
        ),
        # sample006.txt - 贝尔曼方程（矩阵形式）
        (
            "sample006.txt",
            r"""
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
        """,
        ),
        # sample007.txt - 带underbrace的复杂公式
        (
            "sample007.txt",
            r"""
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
        """,
        ),
        # sample008.txt - Q学习公式
        (
            "sample008.txt",
            r"""
        Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
        """,
        ),
        # sample009.txt - 概率分布
        (
            "sample009.txt",
            r"""
        P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
        """,
        ),
        # sample010.txt - 神经网络损失函数
        (
            "sample010.txt",
            r"""
        \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
        """,
        ),
    ]

    print(f"在 {sample_folder} 创建示例文件:")

    for filename, content in samples:
        file_path = os.path.join(sample_folder, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"✓ 创建: {filename}")

    print(f"\n共创建了 {len(samples)} 个示例文件")
    return sample_folder


# 检查安装状态
def check_latex_installation():
    """检查LaTeX安装状态"""
    print("检查LaTeX安装状态...")

    commands_to_check = [
        ("latex", ["latex", "--version"]),
        ("dvipng", ["dvipng", "--version"]),
        ("gs", ["gs", "--version"]),
    ]

    all_ok = True
    for name, cmd in commands_to_check:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✓ {name} 已安装")
            else:
                print(f"✗ {name} 未正确安装")
                all_ok = False
        except FileNotFoundError:
            print(f"✗ {name} 未找到")
            all_ok = False
        except Exception as e:
            print(f"✗ {name} 检查失败: {e}")
            all_ok = False

    return all_ok


# 主程序
if __name__ == "__main__":
    # 检查LaTeX安装
    check_latex_installation()

    # 创建示例文件
    print("\n" + "=" * 60)
    print("创建示例文件")
    print("=" * 60)
    # sample_folder = create_sample_files("latex_formulas")
    sample_folder = "latex_formulas"

    # 创建转换器
    converter = LatexToImage(dpi=300, fontsize=12)

    # 处理所有sample文件
    print("\n" + "=" * 60)
    print("开始批量转换")
    print("=" * 60)

    output_folder = "latex_formulas/output"
    results = converter.process_folder(sample_folder, output_folder)

    # 生成报告
    print("\n" + "=" * 60)
    print("生成处理报告")
    print("=" * 60)

    report = converter.generate_report(results)
    print(report)

    # 显示最终统计
    success_count = sum(1 for r in results if r["success"])
    print(f"\n🎉 处理完成！")
    print(f"📁 输入文件夹: {sample_folder}")
    print(f"📁 输出文件夹: {output_folder}")
    print(f"📊 成功转换: {success_count}/{len(results)} 个文件")
