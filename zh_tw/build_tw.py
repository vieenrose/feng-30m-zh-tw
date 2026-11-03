#!/usr/bin/env python
"""feng-30m zh-CN -> zh-TW. Weight-space program on the TIED embedding matrix.

Only ONE tensor is numerically edited: model.embed_tokens.weight [16384,448] f32, which this
checkpoint uses as both input embedding and unembedding (tie_word_embeddings=true, no
lm_head.weight; core/feng_model.c requires the tie).

R1  Only S_excl glyphs need converting: S_excl = chars that never occur in Traditional text
    (Unihan kTraditionalVariant domain minus the kSimplifiedVariant domain). A char outside
    S_excl is already a legal Traditional glyph, so it keeps identity -- that is why 家 stays
    家 and Unihan's gloss-scoped 傢 suggestion is refused.
R2  Polysemy (发/發/髮, 干/乾/幹, 里/裏/裡, ...) is resolved by the model's own words:
    for every vocab token w containing s, score candidate t by cos(E[w], E[w with s->t]).
    Sum of cosines over the model's real multi-char tokens decides; Unihan kTotalScore is
    the tie-break. No dictionary converter is consulted.
R3  Per Simplified char s with target t (string collision-free, so no id ever decodes to ""):
      swap the piece strings of i_s and i_t, then
      E[i_s] <- v        v = |E[i_s]| * unit(a*unit(E[i_s]) + (1-a)*unit(E[i_t]))
      E[i_t] <- L * v    a = 1 - 0.5*clamp((cos(E[i_s],E[i_t])-0.4)/0.4,0,1)
    i_s (the id the model actually uses for that word) becomes the Traditional glyph and is
    the emittable id. i_t keeps the same word vector, scaled by L, and takes over the
    Simplified string: Simplified input still routes to the word, at L x the logit, so it is
    rarely emitted. L=1 => bilingual input with Simplified leakage; L=0 => pure zh-TW,
    Simplified input falls back to byte tokens.
R4  s with no token for t: i_s takes the string t (row untouched), a donated id carries the
    Simplified string with row L*v.
R5  Multi-char tokens rename their S_excl chars in place; merge rules rename consistently.
"""
import argparse, hashlib, json, os, sys
import numpy as np
from safetensors.numpy import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/luigi/feng-30m"
V = 16384
BAR = chr(124)
SPECIAL_STR = {"".join([BAR, "im_start", BAR]), "".join([BAR, "im_end", BAR]),
               "".join([BAR, "eot", BAR]), "".join([BAR, "pad", BAR])}

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
B2U = bytes_to_unicode(); U2B = {v: bytes([k]) for k, v in B2U.items()}
def pstr(p):
    try: b = b"".join(U2B[ch] for ch in p)
    except KeyError: return None
    try: return b.decode("utf-8")
    except UnicodeDecodeError: return None
def enc(s): return "".join(B2U[b] for b in s.encode("utf-8"))
def is_han(c):
    o = ord(c); return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""): h.update(blk)
    return h.hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=ROOT)
ap.add_argument("--out", default="/home/luigi/feng-30m-zh-tw")
ap.add_argument("--shadow-L", type=float, default=1.0, dest="L")
ap.add_argument("--tag", default="")
ap.add_argument("--dry-run", action="store_true")
A = ap.parse_args()
# ---- the whole transformation is these four knobs -------------------------------
ATTEST_ALL = False   # True = drop every rename the zh-TW corpus never attests (measured worse)
ATTEST = True        # a rename survives only if its Traditional spelling occurs in the zh-TW corpus
CHAR_ATTEST = True   # don't strengthen a Traditional glyph the zh-TW corpus never uses
WORD_COS_MIN = 0.78  # cos(w, its Traditional twin token) needed to treat them as one word (swap)
RENAME_NRM_MIN = 1.05  # carrier floor: a word row must be strong enough to carry a new spelling
SWAP_LEAK_ROW = 'partner'  # 'self' = a swapped-off Simplified string keeps its own row; 'partner' = take the row trained on that string
TWIN_STRENGTH = 0.20   # row transfer into the Traditional twin when the swap gate blocks the swap
CHAR_TWIN = 0.40       # share of the Simplified twin's direction given to a Traditional glyph row
CHAR_GATE = 0.70       # char rows need a looser gate than word swaps (0.66 -> 493 tx, 0.75 -> worse)
FULL_SUPPRESS = True   # every id that can emit a Simplified-only glyph is shrunk
SUPPRESS_L = 0.32      # shrink factor for those ids (purity-bound: 1.00 emits 记)
TRAD_BOOST_REN = 0.70  # the transform's OWN renames are Simplified-trained rows carrying a Traditional spelling: shrink them (mirror of suppression) so the parent's native Traditional rows compete
TRAD_BOOST = 1.10      # mirror

G = json.load(open(f"{HERE}/data/char_graph.json"))
M = json.load(open(f"{HERE}/data/mapping.json"))
s2t, t2s, score = G["s2t"], G["t2s"], G["score"]
S_EXCL = set(G["s_exclusive"])
toks = json.load(open(f"{A.base}/tokenizer.json"))
ov = dict(toks["model"]["vocab"])
inv = {i: p for p, i in ov.items()}
Ss = {i: pstr(inv.get(i, "")) for i in range(V)}
S = {c: i for i in range(V) if Ss[i] and len(Ss[i]) == 1 and is_han(Ss[i]) for c in [Ss[i]]}
ED = load_file(f"{A.base}/model.safetensors")
E = ED["model.embed_tokens.weight"]; assert E.shape == (V, 448) and E.dtype == np.float32
nrm = np.linalg.norm(E.astype(np.float64), axis=1)
unit = E.astype(np.float64) / np.maximum(nrm, 1e-12)[:, None]
cos = lambda i, j: float(unit[i] @ unit[j])
words = [(i, Ss[i]) for i in range(V) if Ss[i] and len(Ss[i]) > 1 and any(is_han(c) for c in Ss[i])]

# ---- R2 word-level polysemy: let the model's own multi-char tokens decide ----------
def pick(s, cands):
    """Choose the zh-TW form of Simplified char s from the model's weight geometry."""
    inv = [(t, S[t]) for t in cands if t in S]
    if len(inv) == 1:
        t, i_t = inv[0]
        return t, ("token+cos", cos(S[s], i_t) if s in S else None)
    if not inv:
        best = max(cands, key=lambda t: (score.get(t, -10**9), -ord(t)))
        return best, ("unigram-prior", None)
    votes = {t: 0.0 for t, _ in inv}
    for wid, w in words:
        if s not in w: continue
        for t, i_t in inv:
            w2 = w.replace(s, t)
            if w2 in ov:                                   # the model has this Traditional word
                votes[t] += max(0.0, cos(wid, ov[w2]))
    tot = sum(votes.values())
    t = max(votes, key=lambda k: (votes[k], score.get(k, -10**9)))
    src = "word-cos" if tot > 0 else "unigram-prior"
    if src == "unigram-prior": t = max(inv, key=lambda ti: (score.get(ti[0], -10**9)))[0]
    return t, (src, max(votes.values()) if tot > 0 else (cos(S[s], S[t]) if s in S else None))

# ---- the transformation: word-level rename/swap, then suppression --------------------
new = dict(Ss); edits = {}; shadow = set(); LEAK_SRC = {}
# glyphs a zh-TW corpus actually uses are legal output (not Simplified leaks)
ref = open(f'{HERE}/data/zh_tw_reference.txt', encoding='utf-8').read() if os.path.exists(f'{HERE}/data/zh_tw_reference.txt') else ''
SKIP = set(c for c in ref if c in S_EXCL)
gmap = {}
for c in sorted(S_EXCL):                            # glyph map for chars with no own token
    if c in gmap: continue
    t, _ = pick(c, s2t[c])
    if t != c: gmap[c] = t
def conv(w): return "".join(gmap.get(c, c) for c in w)
owner = {}
for i in range(V):
    if new[i] is not None: owner.setdefault(new[i], i)
swapped = set(); shadow = set(); wswap = 0; wfree = 0; wleak = 0
cands = sorted((i for i in range(V) if Ss[i] and conv(Ss[i]) != Ss[i] and len(Ss[i]) > 1),
               key=lambda i: -nrm[i])
for w in cands:
    c = conv(Ss[w])
    k = owner.get(c)
    if k is None:
        if ATTEST and c not in ref and (Ss[w] in ref or ATTEST_ALL):
            wleak += 1; continue                           # corpus says the Simplified form is the zh-TW word
        if nrm[w] < RENAME_NRM_MIN:
            wleak += 1; continue           # junk row: keep its string, gets suppressed
        del owner[new[w]]; new[w] = c; owner[c] = w; wfree += 1; continue
    if k not in swapped and Ss[k] == c and cos(w, k) < WORD_COS_MIN:
        wleak += 1
        if TWIN_STRENGTH > 0 and k not in edits:
            v = unit[w] * TWIN_STRENGTH + unit[k] * (1.0 - TWIN_STRENGTH)   # row knowledge only
            edits[k] = (nrm[k] / max(np.linalg.norm(v), 1e-12)) * v
        continue                                       # model does not treat them as one word
    if k not in swapped and Ss[k] == c:                    # k is a real Traditional word token
        a_ = 1.0
        v = a_ * unit[w] + (1.0 - a_) * unit[k]
        v = (nrm[w] / max(np.linalg.norm(v), 1e-12)) * v
        E2w = v
        del owner[new[w]]; del owner[new[k]]
        new[w], new[k] = c, Ss[w]
        owner[c] = w; owner[Ss[w]] = k
        edits[w] = E2w; edits[k] = A.L * E2w; swapped.add(k); shadow.add(k); wswap += 1
        LEAK_SRC[k] = w        # w's row is the one trained on the string Ss[w] that k now emits
    elif nrm[w] > nrm[k]:                                  # both Simplified tokens want c
        del owner[new[w]]; del owner[new[k]]
        new[w], new[k] = c, Ss[k]
        owner[c] = w; owner[Ss[k]] = k; wswap += 1
    else:
        wleak += 1                                         # loses the string: see demotion pass
ctwin = 0
for s0 in sorted(S_EXCL - SKIP):                        # chars: the model's Simplified twin knows
    t0 = gmap.get(s0)
    if not t0 or t0 not in S or s0 not in S: continue
    i_s, i_t = S[s0], S[t0]
    if cos(i_s, i_t) < CHAR_GATE: continue
    if CHAR_ATTEST and t0 not in ref: continue
    if i_t in edits: continue
    v = unit[i_t] * (1.0 - CHAR_TWIN) + unit[i_s] * CHAR_TWIN
    edits[i_t] = (nrm[i_t] / max(np.linalg.norm(v), 1e-12)) * v; ctwin += 1
print(f"[tw] word-swaps {wswap}  word-renames {wfree}  word-leaks {wleak}  char-twins {ctwin}")
E2 = E.copy()
for i, v in edits.items(): E2[i] = v.astype(np.float32)
assert np.isfinite(E2).all()
piece = {i: (inv[i] if new[i] in (None, Ss[i]) else enc(new[i])) for i in range(V)}
# Repair: an id that cannot own a unique string becomes a dead token -- unreachable row,
# synthetic junk piece (same status as the 517 pieces that are not valid UTF-8 at all).
dead = [i for i in range(4, V) if Ss[i] is None and i not in edits]
owner, deadened, seq = {}, 0, 0
for i in range(V):
    p = piece[i]
    if p not in owner:
        owner[p] = i; continue
    j = owner[p]
    loser = i if nrm[i] < nrm[j] else j
    win = j if loser == i else i
    d = dead[seq]; seq += 1
    piece[loser] = enc("\ufffd" + str(seq))          # unique junk, never decoded in practice
    E2[loser] = E[d]; edits[loser] = E[d]; deadened += 1
    owner[p] = win
assert len(set(piece.values())) == V, f"duplicate pieces: {V - len(set(piece.values()))}"
print(f"[tw] pieces unique; {deadened} unassignable ids turned into dead clones")
toks["model"]["vocab"] = {piece[i]: i for i in range(V)}
kept = []
for m in toks["model"]["merges"]:
    x, y = (m.split(" ") if isinstance(m, str) else m)
    ix, iy, ab = ov.get(x), ov.get(y), ov.get(x + y)
    if ix is None or iy is None or ab is None: continue      # inert rule (never applicable)
    if piece[ix] + piece[iy] == piece[ab]: kept.append(f"{piece[ix]} {piece[iy]}")
live = sum(1 for m in toks["model"]["merges"]
           if (lambda xy: xy[0] in ov and xy[1] in ov and xy[0] + xy[1] in ov)
           ((m.split(" ") if isinstance(m, str) else m)))
toks["model"]["merges"] = kept
print(f"[tw] merges: live in parent {live}, kept after rename {len(kept)} ({100*len(kept)/max(live,1):.1f}%)")

# ---- invariants -------------------------------------------------------------------
unt = [i for i in range(V) if i not in edits]
assert np.array_equal(E2[unt], E[unt]), "untouched rows changed"
assert np.isfinite(E2).all()
for i in range(4): assert piece[i] == inv[i], f"special token {i} renamed"
if FULL_SUPPRESS:                       # shrink every id that can emit a Simplified-only glyph
    hard = S_EXCL - SKIP                          # glyphs no zh-TW text in the corpus uses
    for i in range(V):
        if i < 4: continue
        if not new[i] or not any(c in hard for c in new[i]): continue
        src = LEAK_SRC.get(i, i) if SWAP_LEAK_ROW == 'partner' else i
        edits[i] = SUPPRESS_L * E[src].astype(np.float64); shadow.add(i); E2[i] = edits[i].astype(np.float32)
if TRAD_BOOST != 1.0:
    T_EXCL = set(G["t_exclusive"]); hard2 = S_EXCL - SKIP; nb = 0
    for i in range(4, V):
        s2 = new[i]
        if not s2 or not any(c in T_EXCL for c in s2) or any(c in hard2 for c in s2): continue
        _b = TRAD_BOOST_REN if not any(c in T_EXCL for c in (Ss[i] or "")) else TRAD_BOOST
        edits[i] = _b * edits.get(i, E[i].astype(np.float64))
        E2[i] = edits[i].astype(np.float32); nb += 1
    _rn = sum(1 for i in range(4, V) if new[i] and any(c in T_EXCL for c in new[i])
              and not any(c in hard2 for c in new[i]) and not any(c in T_EXCL for c in (Ss[i] or "")))
    print(f"[tw] TRAD_BOOST {TRAD_BOOST} applied to {nb} native-only emitters, ren-branch {_rn}")
nrm2 = np.linalg.norm(E2.astype(np.float64), axis=1)
shadow_max = A.L * float(nrm.max()) + 1e-6
sids = [i for i in range(V) if new[i] and any(c in S_EXCL for c in new[i])]
full = [i for i in sids if nrm2[i] > shadow_max]
print(f"[tw] Simplified-emitting ids: {len(sids)} shadowed (L-scaled), {len(full)} full-strength "
      f"(word leaks) -> {''.join((new[i] or '') for i in full)[:40]}")
assert len(full) <= wleak + 2, "more Simplified emitters than accounted for"
if A.dry_run: print("[dry-run] nothing written"); sys.exit(0)

OUT = A.out + A.tag
os.makedirs(OUT, exist_ok=True)
ED["model.embed_tokens.weight"] = E2
save_file(ED, f"{OUT}/model.safetensors", metadata={"format": "pt"})
toks["model"]["vocab"] = {piece[i]: i for i in range(V)}
json.dump(toks, open(f"{OUT}/tokenizer.json", "w"), ensure_ascii=False)
for fn in ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"):
    if os.path.exists(f"{A.base}/{fn}"):
        f = open(f"{A.base}/{fn}", encoding="utf-8").read()
        if fn == "config.json": f = f.replace("jiaheng0815/feng-30m", "feng-30m-zh-tw")
        open(f"{OUT}/{fn}", "w", encoding="utf-8").write(f)
man = {"op": "zh-TW native conversion of a zh-CN tied embedding (v3: word rename/swap + char twin transfer + suppression/boost rebalance)", "parent": {"path": A.base,
        "model_sha256": sha256(f"{A.base}/model.safetensors")},
       "tensor": "model.embed_tokens.weight", "shape": [V, 448], "dtype": "float32",
       "role": "tied: embedding AND unembedding (no lm_head; C engine requires the tie)",
       "shadow_scale_L": A.L, "rows_written": len(edits),
       "rows_numerically_changed": int((E2 != E).any(axis=1).sum()),
       "scalars_changed": int((E2 != E).sum()), "rows_untouched_bit_identical": V - len(edits),
       "word_swaps": wswap, "word_renames": wfree, "word_leaks": wleak,
       "char_twin_transfers": ctwin, "merges_kept": len(kept),
       "merges_dropped": live - len(kept),
       "knobs": {"WORD_COS_MIN": WORD_COS_MIN, "RENAME_NRM_MIN": RENAME_NRM_MIN,
                 "ATTEST": ATTEST, "ATTEST_ALL": ATTEST_ALL, "TWIN_STRENGTH": TWIN_STRENGTH,
                 "CHAR_TWIN": CHAR_TWIN, "CHAR_GATE": CHAR_GATE, "CHAR_ATTEST": CHAR_ATTEST,
                 "SUPPRESS_L": SUPPRESS_L, "TRAD_BOOST": TRAD_BOOST, "TRAD_BOOST_REN": TRAD_BOOST_REN},
       "formula": "(1) words: for each Simplified token w with a Traditional spelling c -- if c is an "
                  "existing token and cos(w,k)>=WORD_COS_MIN, swap the strings and serve c with w's row "
                  "(k gets w's string, then suppression); if cos < WORD_COS_MIN, k's row keeps"
                  " TWIN_STRENGTH of w's row; if c is free, rename w to c (needs corpus attestation of c"
                  " and |E[w]| >= RENAME_NRM_MIN). (2) chars: a Traditional glyph row takes CHAR_TWIN of"
                  " its Simplified twin's direction at cos >= CHAR_GATE, skipping glyphs the zh-TW"
                  " corpus never uses. (3) softmax rebalance: ids emitting a Simplified-only glyph go to"
                  " SUPPRESS_L of their row; ids emitting a genuinely Traditional glyph go to TRAD_BOOST.",
       "mapping_source": "Unihan kTraditionalVariant/kSimplifiedVariant (glyph sets) + kTotalScore;"
                         " polysemy by cos over the model's own multi-char tokens; word/glyph"
                         " attestation by an eval-disjoint zh-TW corpus",
       "size_bytes": os.path.getsize(f"{OUT}/model.safetensors"),
}
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1, ensure_ascii=False)
print(f"[out] {OUT} rows={man['rows_written']} changed={man['rows_numerically_changed']} "
      f"scalars={man['scalars_changed']} size={man['size_bytes']}")
