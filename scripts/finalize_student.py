"""Finalize the student: pick checkpoint, add chat template + generation config, export GGUF."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT  # noqa: E402
STU = ROOT / "student"
from paths import LLAMA_BIN as BIN  # noqa: E402
from paths import CONVERT_HF_TO_GGUF as CONVERT  # noqa: E402
from paths import PY  # noqa: E402
from paths import require  # noqa: E402

CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(STU / "chat1" / "step1500"))
    ap.add_argument("--dest", default=str(STU / "final"))
    ap.add_argument("--quants", nargs="*", default=["Q8_0", "Q4_K_M"])
    args = ap.parse_args()
    src, dest = Path(args.model).resolve(), Path(args.dest).resolve()
    # 先做路径校验（不依赖 llama.cpp）：源不存在、或源/目的相同/互相包含时，
    # 旧实现会先 rmtree(dest) 把源删掉再报错（A39）。
    if not src.is_dir():
        print(f"找不到源模型目录：{src}（v3 时代的 student/ 已删除，用 --model 指向现有权重）")
        return 2
    if src == dest or src in dest.parents or dest in src.parents:
        print(f"拒绝：--model 与 --dest 不能相同或互相包含（{src} / {dest}）")
        return 2
    require(CONVERT, "llama.cpp 的 convert_hf_to_gguf.py", "FENG_LLAMA_DIR")
    require(PY, "Python 解释器", "FENG_PY")
    # 复制到临时目录再换入：复制失败不会破坏已有的 dest。
    staged = dest.with_name(dest.name + ".new")
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(src, staged)
    backup = dest.with_name(dest.name + ".bak")
    if backup.exists():
        shutil.rmtree(backup)
    if dest.exists():
        dest.rename(backup)
    try:
        staged.rename(dest)
    except OSError:
        if backup.exists() and not dest.exists():
            backup.rename(dest)
        raise
    if backup.exists():
        shutil.rmtree(backup)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(dest)
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(dest)
    (dest / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": 1, "eos_token_id": 0, "pad_token_id": 3, "do_sample": True,
         "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.15}, indent=2), encoding="utf-8")
    (dest / "MODEL_CARD.md").write_text(
        "# feng-30m\n\n"
        "由 feng-0.8b（Qwen3.5-0.8B 全参数微调，个人开发者 jiaheng 微调版，bf16）蒸馏、**从头训练**的 "
        "30.75M 参数对话模型（Qwen3 架构：8 层 / hidden 448 / 7 头 MHA / FFN 896 / tied embedding，"
        "32k 词表 BPE）。\n\n"
        "- 训练：8k 指令阶段 38M+74M tokens + 原生 32k 长上下文阶段\n"
        "- 对话格式：`<|im_start|>user\\n…<|im_end|>\\n<|im_start|>assistant\\n…<|im_end|>`\n"
        "- 身份：feng（个人开发者 jiaheng 微调的 Qwen）\n"
        "- 已知限制：30M 容量有限，常识/推理较弱，建议 `temperature=0.7, top_p=0.9, repetition_penalty=1.15`\n",
        encoding="utf-8")
    print(f"finalized -> {dest}")

    out = dest / "gguf"
    out.mkdir(exist_ok=True)
    f16 = out / "feng-30m-f16.gguf"
    with open(ROOT / "logs" / "convert_student.log", "w", encoding="utf-8") as f:
        rc = subprocess.run([str(PY), str(CONVERT), str(dest), "--outtype", "f16", "--outfile", str(f16)],
                            stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print("convert failed:", (ROOT / "logs" / "convert_student.log").read_text(
            encoding="utf-8", errors="replace")[-600:])
        return 2
    made = [("f16", round(f16.stat().st_size / 1024**2, 1))]
    for q in args.quants:
        dst = out / f"feng-30m-{q}.gguf"
        with open(ROOT / "logs" / f"quant_student_{q}.log", "w", encoding="utf-8") as f:
            rc = subprocess.run([str(BIN / "llama-quantize.exe"), str(f16), str(dst), q, "8"],
                                stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0 and dst.exists():
            made.append((q, round(dst.stat().st_size / 1024**2, 1)))
    print("gguf:", made)
    (out / "export_summary.json").write_text(json.dumps(
        {"source": str(dest), "files": made}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
