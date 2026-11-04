#!/usr/bin/env python3
"""Render an autoplaying README demo: parent (zh-CN) vs feng-30m-zh-tw, same prompts,
real greedy generations, token-by-token. Simplified-only glyphs (the harness purity
criterion) are painted red, Traditional-only glyphs teal, with live counters."""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

HERE = "/tmp/demo"
SAMP = json.load(open("/tmp/demosamples.json", encoding="utf-8"))
G = json.load(open("/home/luigi/feng-tw-opt/data/char_graph.json"))
REF = open("/home/luigi/feng-tw-opt/data/zh_tw_reference.txt", encoding="utf-8").read()
S_EXCL = set(G["s_exclusive"]) - set(c for c in REF if c in G["s_exclusive"])   # harness criterion
T_EXCL = set(G["t_exclusive"])
PROMPTS = ["請寫一段關於秋天落葉的文字。", "請介紹故宮的歷史。", "太陽為什麼會東升西落？"]

W, H = 900, 470
BG, PANEL, PANEL_B = (16, 18, 22), (24, 27, 33), (48, 53, 62)
WHITE, DIM, RED, TEAL, GREEN = (232, 234, 237), (140, 146, 155), (255, 92, 92), (86, 211, 196), (126, 217, 108)
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MONO = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
F = {s: ImageFont.truetype(CJK, s, index=0) for s in (13, 15, 17, 20, 26)}


def wrap_cjk(d, text, font, maxw):
    lines, cur, x = [], "", 0
    for ch in text:
        if ch == "\n":
            if cur: lines.append(cur)
            cur, x = "", 0
            continue
        w = d.textlength(ch, font=font)
        if x + w > maxw and cur:
            lines.append(cur); cur, x = ch, w
        else:
            cur += ch; x += w
    if cur: lines.append(cur)
    return lines


def draw_panel(d, xy, title, title_col, text, _unused, per_side_w):
    x, y, pw, ph = xy
    d.rounded_rectangle([x, y, x + pw, y + ph], 10, fill=PANEL, outline=PANEL_B, width=1)
    d.text((x + 14, y + 12), title, font=F[15], fill=title_col)
    shown = text
    yy = y + 46
    for line in wrap_cjk(d, shown, F[20], per_side_w - 30):
        xx = x + 14
        for ch in line:
            col = RED if ch in S_EXCL else (TEAL if ch in T_EXCL else WHITE)
            d.text((xx, yy), ch, font=F[20], fill=col)
            xx += d.textlength(ch, font=F[20])
        yy += 30
        if yy > y + ph - 62: break
    simp = sum(1 for c in shown if c in S_EXCL)
    trad = sum(1 for c in shown if c in T_EXCL)
    d.line([x + 14, y + ph - 40, x + pw - 14, y + ph - 40], fill=PANEL_B, width=1)
    d.text((x + 14, y + ph - 32), f"Simplified-only  {simp}", font=F[15],
           fill=(RED if simp else GREEN))
    d.text((x + pw - 14 - d.textlength(f"Traditional-only  {trad}", font=F[15]), y + ph - 32),
           f"Traditional-only  {trad}", font=F[15], fill=(TEAL if trad else DIM))


def clean(t):
    t = t.lstrip("\n")
    return t[11:] if t.startswith("assistant") else t

CAPS = {PROMPTS[0]: (20, 26), PROMPTS[1]: (24, 42), PROMPTS[2]: (24, 20)}


def shown_text(p, side, upto):
    full = clean(SAMP[p][side]["steps"][-1])
    cap = CAPS[p][0 if side == "parent" else 1]
    full = full[:cap] + ("…" if len(full) > cap else "")
    return full[:upto] if upto < len(full) else full


def frame(prompt_idx, side, step):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.text((24, 16), "feng-30m  →  feng-30m-zh-tw", font=F[26], fill=WHITE)
    d.text((24, 52), "same prompt, same greedy decoding · weight-space zh-TW nativization (no fine-tuning)",
           font=F[13], fill=DIM)
    d.text((24, 84), "prompt   " + PROMPTS[prompt_idx], font=F[17], fill=(200, 205, 212))
    p = PROMPTS[prompt_idx]
    ptxt = shown_text(p, "parent", 10 ** 9 if side != "parent" else step)
    ztxt = shown_text(p, "zhtw", 10 ** 9 if side != "zhtw" else step)
    pw = (W - 58) // 2
    draw_panel(d, (24, 118, pw, 300), "parent · jiaheng0815/feng-30m  (zh-CN)", DIM, ptxt, 10 ** 9, pw)
    draw_panel(d, (34 + pw, 118, pw, 300), "feng-30m-zh-tw  (this model)", GREEN, ztxt, 10 ** 9, pw)
    for i in range(len(PROMPTS)):
        c = WHITE if i == prompt_idx else PANEL_B
        d.ellipse([24 + i * 20, H - 26, 34 + i * 20, H - 16], fill=c)
    d.text((W - 250, H - 28), "greedy · temperature 0 · real model output", font=F[13], fill=DIM)
    return img


frames, durations = [], []
for pi, pr in enumerate(PROMPTS):
    for side in ("parent", "zhtw"):
        n = len(shown_text(pr, side, 10 ** 9))
        for k in range(0, n, 2):
            frames.append(frame(pi, side, k + 2)); durations.append(60)
        frames.append(frame(pi, side, 10 ** 9)); durations.append(700)
    frames.append(frame(pi, "hold", 0)); durations.append(1600)
pal = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
pal[0].save(f"{HERE}/demo.gif", save_all=True, append_images=pal[1:], duration=durations,
            loop=0, optimize=True, disposal=2)
png = frames[-1].convert("P", palette=Image.ADAPTIVE, colors=128)
png.save(f"{HERE}/demo.png")
print("gif bytes", os.path.getsize(f"{HERE}/demo.gif"), "frames", len(frames))
