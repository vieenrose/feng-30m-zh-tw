package com.feng.feng30m

import android.content.Context
import java.io.File
import java.io.IOException

/**
 * 会话管理：把 assets 里的模型解包到 filesDir，再交给 native 侧 mmap 加载。
 *
 * 为什么先解包而不是直接读 assets：
 * assets 在 APK 里是压缩包内的一段，拿不到「一个可以 mmap 的连续文件」；
 * 解包一次之后，engine 用 mmap 只读映射，页面由内核按需调入 page cache，
 * 不占 Java 堆，也不会因为文件大而 OOM。
 */
class ChatEngine(private val context: Context) {

    // nativeFree 可能在别的线程（onDispose 的后台线程）执行，句柄必须是 volatile：
    // 否则 UI 线程读到的还是旧值，会拿已释放的句柄继续推理。
    @Volatile private var handle = 0L

    val isReady: Boolean get() = handle != 0L

    companion object {
        /** 默认 32k 容量（q2 KV 154 MB）；低内存设备由 open() 自动回退。 */
        const val CTX = 32768
        /** 32k 分配失败时的保守回退（板端 QAT 工作点）。 */
        const val FALLBACK_CTX = 2048
        const val MAX_NEW = 96
        private const val STAMP_FILE = ".assets.stamp"

        /** 引擎默认线程数：min(4, 可用核数)，与 CLI 口径一致 */
        fun defaultThreads(): Int =
            Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
    }

    private fun modelFile(name: String): File = File(context.filesDir, name)

    /** asset 的未压缩长度；拿不到返回 -1。
     *  注意 openFd 只对**未压缩**的 asset 有效（build.gradle.kts 里已经把它加进
     *  noCompress，但这里仍然兜底，免得换构建配置就崩）。 */
    private fun assetSize(name: String): Long =
        try {
            context.assets.openFd(name).use { it.length }
        } catch (e: Exception) {
            -1L
        }

    /** APK 安装/更新时间：每次升级都会变，比文件长度可靠。
     *  同尺寸换权重（重新导出但 shape 不变）时长度完全一样，旧实现会一直用旧文件。 */
    @Suppress("DEPRECATION")
    private fun appVersionStamp(): Long =
        try {
            context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
        } catch (e: Exception) {
            0L
        }

    /** 解包应该达到的“版本+尺寸”指纹；改变即重新解包。 */
    private fun assetStamp(sizes: List<Long>, names: List<String>): String =
        buildString {
            append(appVersionStamp()).append('|')
            names.forEachIndexed { i, n ->
                append(n).append('=').append(sizes[i]).append(';')
                append(modelFile(n).length()).append(',')
            }
        }

    /**
     * 确保 model.bin / tokenizer.bin 已解包到 filesDir。
     *
     * 解包到 `*.tmp` 校验长度后原子 rename：中途被杀掉不会留下“长度正确但内容
     * 半截”的文件（旧实现直接写最终路径）。
     * @param onProgress 0f..1f
     */
    fun extractAssets(onProgress: (Float) -> Unit) {
        val names = listOf("model.bin", "tokenizer.bin")
        val sizes = names.map { assetSize(it) }
        val stampFile = File(context.filesDir, STAMP_FILE)
        val stamp = assetStamp(sizes, names)
        // 上次完整解包过、文件都还在、APK 版本与 asset 长度都没变 -> 直接复用
        if (stampFile.isFile && names.all { modelFile(it).isFile } &&
            stampFile.readText().trim() == stamp) {
            onProgress(1f)
            return
        }

        val total = sizes.filter { it > 0 }.sum().takeIf { it > 0 } ?: -1L
        var done = 0L
        names.forEachIndexed { i, name ->
            val out = modelFile(name)
            val want = sizes[i]
            val tmp = File(context.filesDir, "$name.tmp")
            context.assets.open(name).use { input ->          // 压缩与否都能读
                tmp.outputStream().use { output ->
                    val buf = ByteArray(1 shl 16)
                    while (true) {
                        val n = input.read(buf)
                        if (n <= 0) break
                        output.write(buf, 0, n)
                        done += n
                        onProgress(
                            if (total > 0) (done.toFloat() / total).coerceIn(0f, 1f)
                            else (i + 0.5f) / names.size
                        )
                    }
                }
            }
            if (want > 0 && tmp.length() != want) {
                tmp.delete()
                throw IOException("$name 解包长度不符：$want -> ${tmp.length()}")
            }
            if (!tmp.renameTo(out)) {
                out.delete()
                if (!tmp.renameTo(out)) {
                    tmp.delete()
                    throw IOException("$name 无法写入 ${out.absolutePath}")
                }
            }
            onProgress(if (total > 0) (done.toFloat() / total).coerceIn(0f, 1f) else (i + 1f) / names.size)
        }
        // 两个文件都落地后才写指纹：中途失败下次仍会重试
        stampFile.writeText(stamp)
    }

    /** 模型是否已随 APK 打包（构建脚本会把 model_export 目录拷进 assets） */
    fun hasBundledModel(): Boolean =
        try {
            context.assets.open("model.bin").close()
            true
        } catch (e: Exception) {
            false
        }

    fun open(ctx: Int = CTX, threads: Int = defaultThreads()): Boolean {
        if (handle != 0L) return true
        handle = FengNative.nativeInit(
            modelFile("model.bin").absolutePath,
            modelFile("tokenizer.bin").absolutePath,
            ctx,
            threads
        )
        // 32k 需要 154 MB 连续 KV；低内存/后台压力大时可能失败。
        // 回退到 2048 保证 App 仍可用，状态栏会显示实际 ctx。
        if (handle == 0L && ctx > FALLBACK_CTX) {
            handle = FengNative.nativeInit(
                modelFile("model.bin").absolutePath,
                modelFile("tokenizer.bin").absolutePath,
                FALLBACK_CTX,
                threads
            )
        }
        return handle != 0L
    }

    fun ask(text: String, maxNew: Int = MAX_NEW, cb: FengNative.StreamCallback): Int {
        val h = handle
        if (h == 0L) return -1
        return FengNative.nativeAsk(h, text, maxNew, cb)
    }

    fun abort() {
        val h = handle
        if (h != 0L) FengNative.nativeAbort(h)
    }

    fun reset() {
        val h = handle
        if (h != 0L) FengNative.nativeReset(h)
    }

    fun memory(): String {
        val h = handle
        return if (h != 0L) FengNative.nativeMemory(h) ?: "（空）" else "（未初始化）"
    }

    fun stats(): String {
        val h = handle
        return if (h != 0L) FengNative.nativeStats(h) ?: "" else ""
    }

    fun close() {
        // 先把句柄置 0（volatile）再释放：之后进来的 ask 直接返回 -1；
        // 已经在 native 里的调用由 JNI 的 closing + 进程锁等待（A19）。
        val h = handle
        if (h != 0L) {
            handle = 0L
            FengNative.nativeFree(h)
        }
    }
}
