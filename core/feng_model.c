/* Parse the model.bin image and resolve tensor pointers.
 *
 * 解析保持逐字段 memcpy（对齐安全，与导出端字节精确一致）。
 * v4.0 起是纯 C23：原来的 std::span 表达内存镜像、std::string_view 查名字、
 * std::array 表示定长表全部换成了裸指针 + 长度 + C 字符串——
 * 这套数据结构本来就是 POD，C++ 只污染了函数签名的语法糖。
 * 仍然零分配、零全局构造。 */
#include "feng.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define K_HEADER_BYTES   44u    /* 9 x u32 + 2 x f32 */
#define K_DIRENT_BYTES   68u    /* name[32] + dtype + shape[4] + offset + nbytes */

static const feng_tensor_t *find_tensor(const feng_model_t *m, const char *name)
{
    for (uint32_t i = 0; i < m->hdr.n_tensors; i++) {
        /* 目录项在第 70 行左右被强制 NUL 结尾，所以 strcmp 与原来的
         * string_view 相等比较语义一致 */
        if (strcmp(m->dir[i].name, name) == 0) return &m->dir[i];
    }
    return NULL;
}

static const void *ptr_of(const feng_model_t *m, const feng_tensor_t *t)
{
    return m->blob + t->offset;
}

/* blk.<layer>.<tensor>，顺序与 feng_layer_t 字段一致 */
static const char *const kLayerTensorNames[11] = {
    "attn_norm.weight", "ffn_norm.weight", "attn_q.weight", "attn_k.weight",
    "attn_v.weight",    "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
    "ffn_gate.weight",  "ffn_up.weight",    "ffn_down.weight",
};

[[nodiscard]] int feng_model_init(feng_model_t *m, const void *data, size_t size)
{
    if (m == NULL || data == NULL || size < K_HEADER_BYTES) return -1;
    const uint8_t *p = (const uint8_t *)data;
    /* header: 9 x u32 + 2 x f32 = 44 bytes (packed, parsed field by field) */
    uint32_t v[9] = {0};
    float fv[2] = {0};
    memcpy(v, p, sizeof(v));
    memcpy(fv, p + 36, sizeof(fv));
    m->hdr.magic = v[0];
    m->hdr.version = v[1];
    m->hdr.n_tensors = v[2];
    m->hdr.n_layers = v[3];
    m->hdr.hidden = v[4];
    m->hdr.n_heads = v[5];
    m->hdr.head_dim = v[6];
    m->hdr.ffn = v[7];
    m->hdr.vocab = v[8];
    m->hdr.rms_eps = fv[0];
    m->hdr.rope_theta = fv[1];
    if (m->hdr.magic != (uint32_t)FENG_MAGIC) return -2;
    /* v4.0 的文件格式只有 version 1；板端旧代码把这行放在 FENG_FP16_WEIGHTS 里，
     * 导致 FP16_WEIGHTS=0 时任意版本都放行。 */
    if (m->hdr.version != 1u) return -11;
    if (m->hdr.n_layers == 0 || m->hdr.n_layers > FENG_MAX_LAYERS) return -3;
    if (m->hdr.n_tensors > FENG_MAX_TENSORS) return -8;
    /* 结构上限：这些数字不是“随便写死”的，它们对应 feng_llm.c 的固定工作区。
     * 旧实现只在 FENG_FP16_WEIGHTS=1 时校验 dtype/shape，板端等于没有校验。 */
    if (m->hdr.hidden == 0 || m->hdr.hidden > FENG_MAX_HIDDEN) return -14;
    if (m->hdr.n_heads == 0 || m->hdr.head_dim == 0 ||
        m->hdr.head_dim > FENG_MAX_HEAD_DIM) return -14;
    if ((uint64_t)m->hdr.n_heads * (uint64_t)m->hdr.head_dim != m->hdr.hidden) return -14;
    /* RoPE 按前后两半配对，奇数 head_dim 会漏掉最后一个维度。 */
    if ((m->hdr.head_dim & 1u) != 0u) return -14;
#if FENG_KV_Q2
    /* q2 的 scale 按 block 覆盖 head_dim：不整除就会漏量化尾部维度。 */
    if ((uint64_t)m->hdr.head_dim % (uint64_t)FENG_KV_Q2_BLOCK != 0u) return -14;
#endif
    if (m->hdr.ffn == 0 || m->hdr.ffn > 65536u ||
        m->hdr.vocab == 0 || m->hdr.vocab > 65535u) return -14;
    /* 非法 eps/theta 会让 rmsnorm/rope 产生 NaN/Inf，必须在加载期拒绝。 */
    if (!isfinite(m->hdr.rms_eps) || m->hdr.rms_eps <= 0.0f) return -14;
    if (!isfinite(m->hdr.rope_theta) || m->hdr.rope_theta <= 1.0f) return -14;
    if (size < K_HEADER_BYTES + (size_t)m->hdr.n_tensors * K_DIRENT_BYTES) return -4;
    /* directory entries are packed at 68 bytes each: name[32], dtype, shape[4], offset, nbytes */
    for (uint32_t i = 0; i < m->hdr.n_tensors; i++) {
        const uint8_t *e = p + K_HEADER_BYTES + (size_t)i * K_DIRENT_BYTES;
        feng_tensor_t *d = &m->dir_storage[i];
        memcpy(d->name, e, 32);
        d->name[31] = 0;
        memcpy(&d->dtype, e + 32, 4);
        memcpy(d->shape, e + 36, 16);
        memcpy(&d->offset, e + 52, 8);
        memcpy(&d->nbytes, e + 60, 8);
        /* 每个 tensor 必须完全落在文件镜像内；否则 ptr_of() 会指向 mmap
         * 窗口外，板端表现为 MMU fault，PC 端表现为越界读。 */
        if (d->offset > (uint64_t)size || d->nbytes > (uint64_t)size - d->offset) return -15;
        /* 已知 dtype 的 nbytes 与 shape 必须自洽。 */
        uint64_t elems = 0;
        if (d->shape[0] && d->shape[1]) {
            elems = (uint64_t)d->shape[0] * (uint64_t)d->shape[1];
            if (d->dtype == FENG_DT_FP16) {
                if (elems > UINT64_MAX / 2u || d->nbytes != elems * 2u) return -15;
            } else if (d->dtype == FENG_DT_Q4) {
                if ((d->shape[1] % 64u) != 0u) return -10;
                const uint64_t row = (uint64_t)(d->shape[1] / 64u) * 34u;
                if ((uint64_t)d->shape[0] > UINT64_MAX / row ||
                    d->nbytes != (uint64_t)d->shape[0] * row) return -15;
            } else {
                return -9;
            }
        } else if (d->shape[0] && d->dtype == FENG_DT_FP16) {
            /* shape[0] 是 u32，乘 2 不可能溢出 u64——旧写法 `shape[0] > UINT64_MAX/2`
             * 恒为假（-Wtype-limits），等于没有检查；真正的检查只有下面这条。 */
            if (d->nbytes != (uint64_t)d->shape[0] * 2u) return -15;
        } else if (d->dtype != FENG_DT_FP16 && d->dtype != FENG_DT_Q4) {
            return -9;
        }
    }
    m->dir = m->dir_storage;
    m->blob = p;

    const feng_tensor_t *t = find_tensor(m, "tok_embd.weight");
    if (t == NULL) return -5;
    if (t->shape[0] != m->hdr.vocab || t->shape[1] != m->hdr.hidden) return -16;
    m->tok_embd = ptr_of(m, t);
    m->tok_embd_dtype = t->dtype;
    t = find_tensor(m, "output_norm.weight");
    if (t == NULL) return -6;
    if (t->shape[0] != m->hdr.hidden || t->dtype != FENG_DT_FP16) return -16;
    m->out_norm = ptr_of(m, t);

    for (uint32_t l = 0; l < m->hdr.n_layers; l++) {
        const void *slots[11] = {0};
        uint32_t dts[11] = {0};
        for (int k = 0; k < 11; k++) {
            char buf[64];
            /* 名字已 NUL 结尾，不需要原来的 %.*s + string_view.size() */
            snprintf(buf, sizeof(buf), "blk.%u.%s", (unsigned)l, kLayerTensorNames[k]);
            const feng_tensor_t *tt = find_tensor(m, buf);
            if (tt == NULL) {
                printf("missing tensor %s\n", buf);
                return -7;
            }
            /* 目录项自身“nbytes 与 shape 自洽”不代表满足推理消费尺寸：
             * 例如 norm 只存 1 个 fp16 也能过上面的校验，但 rmsnorm 会按
             * hidden 读完整段。这里逐项绑定到 header 的实际尺寸。 */
            uint32_t d0 = m->hdr.hidden, d1 = 0;
            int two = 0;
            switch (k) {
            case 2: case 3: case 4: case 5:
                d1 = m->hdr.hidden; two = 1; break;
            case 6: case 7:
                d0 = m->hdr.head_dim; break;
            case 8: case 9:
                d0 = m->hdr.ffn; d1 = m->hdr.hidden; two = 1; break;
            case 10:
                d0 = m->hdr.hidden; d1 = m->hdr.ffn; two = 1; break;
            default: break;                     /* 0/1: attn_norm/ffn_norm */
            }
            if (tt->shape[0] != d0 || (two ? tt->shape[1] != d1 : tt->shape[1] != 0u)) {
                printf("bad tensor shape %s (got %u x %u)\n", buf,
                       (unsigned)tt->shape[0], (unsigned)tt->shape[1]);
                return -16;
            }
            slots[k] = ptr_of(m, tt);
            dts[k] = tt->dtype;
        }
        m->layers[l].attn_norm = slots[0];
        m->layers[l].ffn_norm = slots[1];
        m->layers[l].wq = slots[2];
        m->layers[l].wk = slots[3];
        m->layers[l].wv = slots[4];
        m->layers[l].wo = slots[5];
        m->layers[l].q_norm = slots[6];
        m->layers[l].k_norm = slots[7];
        m->layers[l].gate = slots[8];
        m->layers[l].up = slots[9];
        m->layers[l].down = slots[10];
        /* norm 必须 fp16（rmsnorm 按 uint16_t* 读）。 */
        if (dts[0] != FENG_DT_FP16 || dts[1] != FENG_DT_FP16 ||
            dts[6] != FENG_DT_FP16 || dts[7] != FENG_DT_FP16) return -12;
#if FENG_FP16_WEIGHTS
        /* 同层 7 个矩阵必须同格式：wq/wk/wv 与 gate/up 各共用一次合并 GEMV，
         * 那个 API 只有一个 dtype 参数。 */
        if (dts[3] != dts[2] || dts[4] != dts[2] || dts[5] != dts[2] ||
            dts[8] != dts[2] || dts[9] != dts[2] || dts[10] != dts[2]) return -13;
        m->layers[l].mat_dtype = dts[2];
#else
        /* 板端 FENG_FP16_WEIGHTS=0：FENG_LAYER_DTYPE 固定按 Q4 解包，
         * 所以 7 个矩阵必须真的是 Q4。以前这里不检查，fp16 导出的模型
         * 会被静默按 Q4 读，logits 全错。 */
        static const int mats[7] = {2, 3, 4, 5, 8, 9, 10};
        for (int k = 0; k < 7; k++)
            if (dts[mats[k]] != FENG_DT_Q4) return -13;
#endif
    }
    return 0;
}
