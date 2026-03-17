"""
将数据集格式转成如下 xtuner 格式(https://xtuner.readthedocs.io/zh-cn/latest/get_started/mllm_sft.html):
{"id": 0, "messages": [{"role": "user", "content": "The distance between two stars is 6.52 \u00d7 10^5 light years. What is the distance between the two stars in parsecs? (1 parsec = 3.26 light years)\nAnswer Choices: (A) 2 \u00d7 10^5 (B) 4 \u00d7 10^6 (C) 5 \u00d7 10^7 (D) 7 \u00d7 10^7 (E) 9 \u00d7 10^8"}, {"from": "assistant", "value": "Let's think about the multi-choice question.\n6.52 \u00d7 10^5 ly / (3.26 ly/parsec) = 2 x 10^5 persec\nThe answer is A."}]}
{"id": 1, "messages": [{"role": "user", "content": "How many ways can the letters in the word COMMON be arranged?\nAnswer Choices: (A) 6 (B) 30 (C) 90 (D) 120 (E) 180"}, {"from": "assistant", "value": "Let's solve the multi-choice question step by step.\nAccording to the above the # of permutations of 6 letters COMMON out of which 2 O's and 2 M's are identical is 6!2!\u22172!=180\nThe answer is E."}]}
{"id": 2, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {'url':"resource/mscoco_dog_000000319154.jpg","image_wh": [375, 500]}}, {"type": "text", "text": "<IMG_CONTEXT>\n图片中的狗是什么颜色？"}]}, {"role": "assistant", "content": "图片中的狗是棕色的。"}, {"role": "user", "content": "这只狗有什么类型的项圈？"}, {"role": "assistant", "content": "这只狗有一条红色的项圈。"}, {"role": "user", "content": "那只狗在窗后的位置是什么？"}, {"role": "assistant", "content": "那只狗正坐在窗户后面。"}, {"role": "user", "content": "那只狗望着窗外时，它的表情是怎样的？"}, {"role": "assistant", "content": "当狗凝视窗外的时候，它看起来是在渴望着什么东西。 这也许暗示了它想要到外面去或者与某人或某物进行互动。"}, {"role": "user", "content": "你能描述一下狗的毛吗？"}, {"role": "assistant", "content": "这只狗有蓬松、毛茸茸的棕色皮毛，使它看起来既可爱又温顺。"}]}
{"id": 3, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {'url':"resource/mscoco_dog_000000319154.jpg","image_wh": [375, 500]}}, {"type": "text", "text": "<IMG_CONTEXT>\n图片中的狗是什么颜色？"}]}, {"role": "assistant", "content": "图片中的狗是棕色的。"}]}
"""

import json
import os
from PIL import Image

# 1. 定义路径
dataset_root = "/root/data/dataset/VLM-formula-recognition-dataset_intern_camp"
# 图片所在的根目录，对应 vl.py 中的 media_root
media_root = os.path.join(dataset_root, "train")

input_file = os.path.join(dataset_root, "train/train_mini.jsonl")
output_file = os.path.join(dataset_root, "train/train_mini_xt.jsonl")

print(f"Processing file: {input_file}")

with (
    open(input_file, "r", encoding="utf-8") as fin,
    open(output_file, "w", encoding="utf-8") as fout,
):
    for i, line in enumerate(fin):
        data = json.loads(line)

        new_entry = {"id": i, "messages": []}

        # 获取图片路径
        image_path = (
            data["images"][0] if "images" in data and len(data["images"]) > 0 else None
        )

        # 获取图片尺寸 (W, H)
        image_wh = [0, 0]
        if image_path:
            # 修正：图片路径应该是相对于 media_root 的
            full_img_path = os.path.join(media_root, image_path)
            try:
                if os.path.exists(full_img_path):
                    with Image.open(full_img_path) as img:
                        image_wh = list(img.size)
                else:
                    print(f"Warning: Image not found at {full_img_path}")
            except Exception as e:
                print(f"Error reading image {full_img_path}: {e}")

        for msg in data["messages"]:
            if msg["role"] == "user":
                # --- 关键修复开始 ---
                # 1. 先把原来的 <image> 删掉，防止重复或位置不对
                clean_text = msg["content"].replace("<image>", "").strip()

                new_content_list = []

                if image_path:
                    # 2. 如果有图片，必须在 text 字段里手动加上 <IMG_CONTEXT> 标签！
                    # XTuner 需要这个标签来定位图片 embedding 的插入位置
                    # 注意：XTuner 默认使用 <IMG_CONTEXT> 作为占位符，而不是 <image>
                    final_text = f"<IMG_CONTEXT>\n{clean_text}"

                    new_content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_path,  # 这里保持相对路径，XTuner 会结合 media_root 使用
                                "image_wh": image_wh,
                            },
                        }
                    )
                else:
                    # 如果没图片，就不加标签
                    final_text = clean_text

                # 添加文本部分
                new_content_list.append({"type": "text", "text": final_text})
                # --- 关键修复结束 ---

                new_entry["messages"].append(
                    {"role": "user", "content": new_content_list}
                )

            else:
                new_entry["messages"].append(
                    {"role": "assistant", "content": msg["content"]}
                )

        fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

print(f"Done! Output saved to: {output_file}")
