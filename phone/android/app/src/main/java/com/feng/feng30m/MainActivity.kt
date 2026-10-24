package com.feng.feng30m

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

enum class Role { USER, BOT, NOTE }

data class Msg(val role: Role, val text: String, val meta: String = "")

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { FengTheme { FengApp() } }
    }
}

@androidx.compose.runtime.Composable
fun FengApp() {
    val context = LocalContext.current
    val engine = remember { ChatEngine(context.applicationContext) }
    val main = remember { Handler(Looper.getMainLooper()) }
    val scope = androidx.compose.runtime.rememberCoroutineScope()

    val messages = remember { mutableStateListOf<Msg>() }
    var ready by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("加载中…") }
    var backend by remember { mutableStateOf("") }
    var loadError by remember { mutableStateOf<String?>(null) }

    // 生命周期所有权：界面销毁时释放 native 会话（旧实现从不 close，反复重开
    // 会累积几百 MB 的映射/KV）。先 abort 再在后台 close：close 会等 native
    // 的 in-flight 推理退出，放在主线程会 ANR。
    DisposableEffect(Unit) {
        onDispose {
            engine.abort()
            Thread {
                try {
                    engine.close()
                } catch (_: Throwable) {
                }
            }.start()
        }
    }

    // 启动：解包 assets -> nativeInit
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            try {
                if (!engine.hasBundledModel()) {
                    main.post {
                        loadError = "APK 里没有模型文件（assets/assets 为空）。\n" +
                            "请用 phone/tools/build_apk.ps1 构建，它会把 model.bin / tokenizer.bin 打进包里。"
                    }
                    return@withContext
                }
                engine.extractAssets { p ->
                    main.post { status = "解包模型 ${(p * 100).toInt()}%" }
                }
                main.post { status = "加载模型…" }
                val ok = engine.open()
                main.post {
                    if (ok) {
                        ready = true
                        backend = engine.stats()
                        status = "就绪"
                    } else {
                        loadError = "nativeInit 失败：模型文件损坏或内存不足"
                    }
                }
            } catch (e: Throwable) {
                main.post { loadError = "启动失败：${e.message}" }
            }
        }
    }

    fun send(text: String) {
        if (!ready || busy || text.isBlank()) return
        messages.add(Msg(Role.USER, text))
        val bot = Msg(Role.BOT, "")
        messages.add(bot)
        val idx = messages.lastIndex
        busy = true

        scope.launch {
            val sb = StringBuilder()
            // onDone 是唯一的正常结束契约（JNI 失败路径也会回调它）。再加一个原子
            // 标志兜底：nativeAsk 抛异常或返回负数时 UI 也必须解除 busy，
            // 否则输入框永久禁用（A16）。
            val done = AtomicBoolean(false)
            // 生成期间用户可能点了「清空」，消息列表会被清掉——下标失效就丢弃回调，
            // 不要往一个已经不存在的槽位写。
            fun put(text: String, meta: String = "") {
                if (idx <= messages.lastIndex) messages[idx] = Msg(Role.BOT, text, meta)
            }
            fun finish(text: String, meta: String) {
                if (!done.compareAndSet(false, true)) return
                main.post {
                    put(text, meta)
                    status = engine.stats()
                    backend = engine.stats()
                    busy = false
                }
            }
            val cb = object : FengNative.StreamCallback {
                override fun onToken(token: String) {
                    sb.append(token)
                    val snapshot = sb.toString()
                    main.post { put(snapshot) }
                }

                override fun onDone(genTokens: Int, decodeMs: Long) {
                    val tps = if (decodeMs > 0) genTokens * 1000.0 / decodeMs else 0.0
                    val meta = if (genTokens == 0) "tool 秒回" else
                        "$genTokens tok ｜ ${"%.1f".format(tps)} tok/s"
                    finish(sb.toString(), meta)
                }
            }
            // 推理跑在 Default 线程池上；native 侧是阻塞调用
            var rc = -1
            try {
                withContext(Dispatchers.Default) {
                    rc = engine.ask(text, ChatEngine.MAX_NEW, cb)
                }
            } catch (e: Throwable) {
                finish("${sb}（出错：${e.message}）", "error")
            } finally {
                if (rc < 0) finish("${sb}（未初始化）", "error")
            }
        }
    }

    ChatScreen(
        messages = messages,
        ready = ready,
        busy = busy,
        status = status,
        backend = backend,
        loadError = loadError,
        onSend = ::send,
        onStop = { engine.abort() },
        // reset/memory 会拿 native 的进程锁：生成期间在主线程调用要等整轮
        // prefill+decode（ANR，A20）。放到后台线程执行；busy 时按钮本身也禁用。
        onReset = {
            if (!busy) {
                scope.launch {
                    withContext(Dispatchers.Default) { engine.reset() }
                    messages.clear()
                    messages.add(Msg(Role.NOTE, "已清空上下文与记忆"))
                    status = engine.stats()
                }
            }
        },
        onMemory = {
            if (!busy) {
                scope.launch {
                    val snap = withContext(Dispatchers.Default) { engine.memory() }
                    messages.add(Msg(Role.NOTE, "记忆：$snap"))
                }
            }
        }
    )
}
