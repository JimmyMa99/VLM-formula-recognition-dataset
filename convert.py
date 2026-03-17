import json, os, sys

in_path = "/root/data/dataset/VLM-formula-recognition-dataset_intern_camp/train/train_mini.jsonl"
out_path = "/root/data/dataset/VLM-formula-recognition-dataset_intern_camp/train/train_mini_abs.jsonl"
base = "/root/data/dataset/VLM-formula-recognition-dataset_intern_camp/train/"


def to_abs(p):
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(base, p)


n = 0
m = 0


with (
    open(in_path, "r", encoding="utf-8") as fin,
    open(out_path, "w", encoding="utf-8") as fout,
):
    for line in fin:
        if not line.strip():
            continue

        obj = json.loads(line)
        changed = False

        if "images" in obj and isinstance(obj["images"], list):
            obj["images"] = [to_abs(x) for x in obj["images"]]
            changed = True
        if "messages" in obj:
            for msg in obj["messages"]:
                c = msg.get("content")
                if isinstance(c, list):
                    for part in c:
                        if part.get("type") == "image" and "image" in part:
                            part["image"] = to_abs(part["image"])
                            changed = True

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        n += 1
        m += changed


print(f"processed lines={n}, patched={m}, saved -> {out_path}")
