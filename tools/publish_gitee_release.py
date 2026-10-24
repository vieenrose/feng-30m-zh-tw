#!/usr/bin/env python
"""在 Gitee 上创建发行版（Release）并上传附件。

    set GITEE_TOKEN=xxxxxxxx            # PowerShell: $env:GITEE_TOKEN="xxxx"
    python tools\\publish_gitee_release.py --tag v4.0 --notes-file notes.md

令牌放哪：
  1. 环境变量 `GITEE_TOKEN`（推荐）
  2. `scripts/local_paths.json` 的 `gitee_token` 键（该文件已 gitignore）
  **不要把令牌写进本文件**——本文件是入库的。

为什么单独写这个脚本（而不是塞进 tools/publish_all.py）：
  `publish_all.py` 内嵌了魔搭令牌、已 gitignore；Gitee 的令牌不该跟着它一起隐身，
  因为**创发行版**是有额度、有公开影响的操作，值得显式跑一次。

Gitee 的两条硬限制（2026-10-17 实测）：
  - **单个附件 ≤ 100 MB**（免费账号），超了 API 直接拒
  - 单仓库附件总量 ≤ 1 GB
  所以 PC 包（118 MB）传不上去，只传三个板端/引擎包；PC 权重在说明里指向 Hugging Face
  与魔搭。
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
API = "https://gitee.com/api/v5"
OWNER_REPO = "fengjiaheng123/feng-30m"
MAX_ATTACH = 100 * 1024 * 1024          # Gitee 免费账号单附件上限

# 超过这条线的包不传（PC 权重 118 MB，走 HF / 魔搭）
SKIP_OVER = MAX_ATTACH


def load_token() -> str:
    """环境变量 → local_paths.json → **git 凭据管理器**。

    最后那条是实测出来的：Gitee 的凭据管理器里存的那串密码本身就是
    **32 位十六进制的私人令牌**（不是账号密码），可以直接当 access_token 用。
    这样只要 `git push gitee` 能成功，发布脚本就能跑，不用另外再配一份令牌。
    """
    tok = os.environ.get("GITEE_TOKEN") or os.environ.get("GITEE_ACCESS_TOKEN")
    if tok:
        return tok.strip()
    lp = ROOT / "scripts" / "local_paths.json"
    if lp.exists():
        try:
            tok = json.loads(lp.read_text(encoding="utf-8")).get("gitee_token")
            if tok:
                return tok.strip()
        except Exception:
            pass
    try:
        import subprocess
        r = subprocess.run(["git", "credential", "fill"], cwd=ROOT, text=True,
                           capture_output=True, timeout=15,
                           input="protocol=https\nhost=gitee.com\n\n")
        for line in r.stdout.splitlines():
            if line.startswith("password="):
                pw = line.split("=", 1)[1].strip()
                # 只接受形如 32 位十六进制的私人令牌，避免把账号密码当令牌用
                if len(pw) == 32 and all(c in "0123456789abcdef" for c in pw.lower()):
                    return pw
    except Exception:
        pass
    return ""


def post(url: str, data: bytes, headers: dict) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body


def get(url: str, token: str) -> tuple[int, dict | list | str]:
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(f"{url}{sep}access_token={urllib.parse.quote(token)}")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def form_body(fields: dict[str, str]) -> tuple[str, bytes]:
    boundary = "----feng30m" + os.urandom(8).hex()
    out = b""
    for k, v in fields.items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n"
                f"{v}\r\n").encode("utf-8")
    out += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", out


def file_body(field: str, path: Path) -> tuple[str, bytes]:
    boundary = "----feng30m" + os.urandom(8).hex()
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: {ctype}\r\n\r\n").encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", head + path.read_bytes() + tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4.0")
    ap.add_argument("--name", default="")
    ap.add_argument("--notes-file", default="")
    ap.add_argument("--dist", default="dist")
    ap.add_argument("--target", default="main")
    ap.add_argument("--skip-upload", action="store_true", help="只建发行版，不传附件")
    a = ap.parse_args()

    token = load_token()
    if not token:
        print("✗ 没有 Gitee 令牌。设置环境变量 GITEE_TOKEN，或写进 "
              "scripts/local_paths.json 的 gitee_token。")
        print("  生成：Gitee → 头像 → 设置 → 安全设置 → 私人令牌 → 生成（勾 projects）")
        return 2

    body = Path(a.notes_file).read_text(encoding="utf-8") if a.notes_file else ""
    name = a.name or f"feng-30m {a.tag.lstrip('v')}"

    # 1) 建发行版（已存在就拿现成的）
    fields = {"tag_name": a.tag, "name": name, "body": body,
              "target_commitish": a.target, "prerelease": "false"}
    ctype, data = form_body({**fields, "access_token": token})
    code, resp = post(f"{API}/repos/{OWNER_REPO}/releases", data,
                      {"Content-Type": ctype})
    if code in (200, 201):
        rid = resp["id"] if isinstance(resp, dict) else None
        print(f"✓ 发行版已创建：{resp.get('html_url') if isinstance(resp, dict) else ''}")
    else:
        print(f"建发行版返回 {code}：{str(resp)[:200]}")
        code2, lst = get(f"{API}/repos/{OWNER_REPO}/releases", token)
        rid = None
        if code2 == 200 and isinstance(lst, list):
            for rel in lst:
                if rel.get("tag_name") == a.tag:
                    rid = rel["id"]
                    print(f"  已存在同名发行版：{rel.get('html_url')}")
                    break
        if rid is None:
            return 1

    if a.skip_upload:
        return 0

    # 2) 传附件（>100 MB 的跳过并说明）
    dist = (ROOT / a.dist).resolve()
    files = sorted(dist.glob(f"*{a.tag}*.zip"))
    if not files:
        print(f"  ✗ {dist} 里没有 {a.tag} 的 zip")
        return 1
    failed = []
    for f in files:
        size = f.stat().st_size
        if size > SKIP_OVER:
            print(f"  跳过 {f.name}（{size/1048576:.1f} MB > Gitee 上限 100 MB）")
            continue
        ctype, data = file_body("file", f)
        url = (f"{API}/repos/{OWNER_REPO}/releases/{rid}/attach_files"
               f"?access_token={urllib.parse.quote(token)}")
        code, resp = post(url, data, {"Content-Type": ctype})
        if code in (200, 201):
            print(f"  ✓ {f.name}（{size/1048576:.2f} MB）")
        else:
            print(f"  ✗ {f.name}：{code} {str(resp)[:160]}")
            failed.append(f.name)

    code, rel = get(f"{API}/repos/{OWNER_REPO}/releases/tags/{a.tag}", token)
    if code == 200 and isinstance(rel, dict):
        print(f"\n完成：{rel.get('html_url')}")
    # 附件失败必须非零退出：否则自动化会把“release 缺附件”当成发布成功（A47）。
    if failed:
        print(f"\n✗ {len(failed)} 个附件上传失败：{', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
