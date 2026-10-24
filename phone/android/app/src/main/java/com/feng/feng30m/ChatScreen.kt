package com.feng.feng30m

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    messages: List<Msg>,
    ready: Boolean,
    busy: Boolean,
    status: String,
    backend: String,
    loadError: String?,
    onSend: (String) -> Unit,
    onStop: () -> Unit,
    onReset: () -> Unit,
    onMemory: () -> Unit,
) {
    var input by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    // 有新内容就滚到底
    LaunchedEffect(messages.size, messages.lastOrNull()?.text?.length) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.size - 1)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("feng-30m") },
                actions = {
                    TextButton(onClick = onMemory, enabled = ready && !busy) { Text("记忆") }
                    TextButton(onClick = onReset, enabled = ready && !busy) { Text("清空") }
                }
            )
        },
        bottomBar = {
            Column(Modifier.navigationBarsPadding().imePadding()) {
                if (backend.isNotBlank()) {
                    Text(
                        backend,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 2.dp),
                        textAlign = TextAlign.Center,
                    )
                }
                Row(
                    Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    OutlinedTextField(
                        value = input,
                        onValueChange = { input = it },
                        modifier = Modifier.weight(1f),
                        placeholder = { Text(if (ready) "说点什么…" else status) },
                        enabled = ready && !busy,
                        maxLines = 4,
                    )
                    IconButton(
                        onClick = {
                            if (busy) onStop() else { onSend(input); input = "" }
                        },
                        enabled = ready,
                    ) {
                        Icon(
                            imageVector = if (busy) Icons.Filled.Stop else Icons.Filled.Send,
                            contentDescription = if (busy) "停止" else "发送",
                        )
                    }
                }
            }
        },
    ) { pad ->
        Box(Modifier.fillMaxSize().padding(pad)) {
            when {
                loadError != null -> Column(
                    Modifier.fillMaxSize().padding(24.dp),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text("无法启动", style = MaterialTheme.typography.titleLarge)
                    Text(loadError, modifier = Modifier.padding(top = 12.dp))
                }

                !ready -> Column(
                    Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    CircularProgressIndicator()
                    Text(status, modifier = Modifier.padding(top = 16.dp))
                }

                else -> LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    itemsIndexed(messages) { _, m -> Bubble(m) }
                }
            }
        }
    }
}

@Composable
private fun Bubble(m: Msg) {
    val isUser = m.role == Role.USER
    val bg = when (m.role) {
        Role.USER -> MaterialTheme.colorScheme.primaryContainer
        Role.BOT -> MaterialTheme.colorScheme.surfaceVariant
        Role.NOTE -> Color.Transparent
    }
    val align = if (isUser) Alignment.CenterEnd else Alignment.CenterStart
    Box(Modifier.fillMaxWidth(), contentAlignment = align) {
        Column(horizontalAlignment = if (isUser) Alignment.End else Alignment.Start) {
            if (m.role == Role.NOTE) {
                Text(
                    m.text,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    textAlign = TextAlign.Center,
                )
            } else {
                Surface(
                    color = bg,
                    shape = RoundedCornerShape(14.dp),
                    modifier = Modifier.widthIn(max = 320.dp),
                ) {
                    Text(
                        text = m.text.ifEmpty { "…" },
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                        style = MaterialTheme.typography.bodyLarge,
                    )
                }
                if (m.meta.isNotBlank()) {
                    Text(
                        m.meta,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontFamily = FontFamily.Monospace,
                        modifier = Modifier.padding(top = 2.dp, start = 4.dp, end = 4.dp),
                    )
                }
            }
        }
    }
}
