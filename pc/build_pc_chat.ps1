# 编译 PC 端引擎运行时（带 tool）。
#
# **v4.0 起是两趟构建**：核心是纯 C23，驱动是 C++23，两者标准互斥、必须分开编译。
#   - 实测 MSVC：一条命令里同时给 /std:clatest 和 /std:c++latest 会报
#     `error D8016: 选项不兼容`；只给 /std:clatest 则 C++ 驱动挂（<span> 需要 C++20+）。
#   - MinGW 同理：g++ 会把 .c 当 C++ 编，所以核心走 gcc -std=c23、驱动走 g++ -std=c++23。
#
# 分工（详见根 AGENTS.md）：
#   main\feng_*.c   推理核心，板端与 PC 共用，严格 C23、零堆、无全局构造
#   pc\*.cpp        PC 侧驱动/套件，C++23（std::chrono / std::span / 花括号初始化等）
#   main\feng_cuda.cu  GPU 后端，nvcc（无 C 模式，永久留 C++），见 build_pc_cuda.ps1
#
# Windows 默认 MSVC：
#   .\build_pc_chat.ps1
# 备用路径（MinGW gcc/g++）：
#   .\build_pc_chat.ps1 -MinGW
#
# 产物（两种工具链输出同名 exe）：
#   pc_chat_q2b8 / pc_kv_suite_q2b8 / pc_kv_suite_i8 / pc_mt_suite / pc_check / pc_bench
#   pc_calc_test / pc_tools_test / pc_mem_test / pc_sample_test
#   pc_calc_ask / pc_mem_ask / pc_rand_ask / pc_time_ask
param([switch]$MinGW)
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    # 共用推理核心（C23）
    $core = @(
        "..\core\feng_model.c", "..\core\feng_llm.c", "..\core\feng_quant.c",
        "..\core\feng_smp.c", "..\core\feng_tokenizer.c", "..\core\feng_sample.c",
        "..\core\feng_calc.c", "..\core\feng_tools.c", "..\core\feng_memory.c"
    )
    # PC 侧驱动（C++23）
    $q2Drivers = @(
        "pc_chat.cpp", "pc_kv_suite.cpp", "pc_mt_suite.cpp",
        "pc_bench.cpp",
        "pc_calc_test.cpp", "pc_tools_test.cpp", "pc_mem_test.cpp",
        "pc_sample_test.cpp", "pc_calc_ask.cpp", "pc_mem_ask.cpp",
        "pc_rand_ask.cpp", "pc_time_ask.cpp"
    )

    function Quote-Arg([string]$a) {
        # Windows 命令行给 argv 的规则：参数里的引号要反斜杠转义；**结尾的反斜杠**
        # 必须成对，否则会把结束引号吃掉。踩过的坑：`/FoE:\...\q2\` 原样加引号后
        # 变成 `/FoE:\...\q2"`，cl 报 D8036「/Fo 不能用于多个源文件」，默认 MSVC
        # 构建直接失败（只有手工 -MinGW 能过）。
        if ($a.Length -gt 0 -and $a -notmatch '[\s"]') { return $a }
        $s = [regex]::Replace($a, '(\\*)"', '$1$1\"')
        $s = [regex]::Replace($s, '(\\+)$', '$1$1')
        return '"' + $s + '"'
    }

    function Invoke-Native([string]$exe, [string[]]$argv) {
        # 用 System.Diagnostics.Process 而不是 `& $exe | ForEach-Object`：
        # 后者在某些非控制台宿主（CI/代理沙箱里的 PowerShell）会对原生 exe 报
        # 「无法在管道中间运行文档」直接崩掉；Process.Start 是纯托管机制，
        # 任何宿主里行为一致，输出/退出码也照拿。
        # 参数自己按 Windows 规则加引号：ArgumentList 在 Windows PowerShell 5.1
        # （check_all 优先用的那个）里不存在。
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $exe
        $psi.Arguments = (($argv | ForEach-Object { Quote-Arg $_ }) -join ' ')
        $psi.WorkingDirectory = (Get-Location).Path   # 继承脚本的 Push-Location
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $p = [System.Diagnostics.Process]::Start($psi)
        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()
        $p.WaitForExit()
        if ($stdout) { Write-Host $stdout }
        if ($stderr) { Write-Host $stderr }
        return $p.ExitCode
    }

    if (-not $MinGW) {
        # ---------------- MSVC（C23 核心 + C++23 驱动，两趟编译） ----------------
        if ($env:VSCMD_VER) { } else {
            $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
            if (-not (Test-Path $vswhere)) {
                throw "找不到 vswhere（需要 Visual Studio Build Tools）；或改用 -MinGW"
            }
            $vsPath = & $vswhere -latest -products * -property installationPath | Select-Object -First 1
            $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
            if (-not (Test-Path $vcvars)) { throw "找不到 vcvars64.bat：$vcvars" }
            $lines = & cmd /c "`"$vcvars`" >nul 2>&1 && set"
            foreach ($line in $lines) {
                $i = $line.IndexOf('=')
                if ($i -gt 0) { Set-Item -Path ("env:" + $line.Substring(0, $i)) -Value $line.Substring($i + 1) }
            }
            Write-Host "MSVC 环境：$vsPath" -ForegroundColor DarkGray
        }
        $objRoot = Join-Path $PSScriptRoot "msvc_obj"
        $objQ2 = Join-Path $objRoot "q2"
        $objFp32 = Join-Path $objRoot "fp32"
        $objI8 = Join-Path $objRoot "i8"
        New-Item -ItemType Directory -Force -Path $objQ2, $objFp32, $objI8 | Out-Null

        # 核心走 C23：/TC 强制按 C 编译，/std:clatest 打开 C23（[[nodiscard]] 等要它）
        # FENG_FP16_WEIGHTS 必须同时给核心与驱动：它改变 feng_layer_t 的大小，
        # 两边不一致就是 ABI 错位（结构体越界读写）。
        # FENG_USE_OMP 也必须同时给核心：GEMV 的 `#pragma omp parallel for` 在
        # feng_quant.c（用 $cFlags 单独编译的那趟）里——只给驱动开 OpenMP
        # 等于什么都没并行（A52）。
        $cFlags = @("/nologo", "/c", "/O2", "/std:clatest", "/TC", "/utf-8", "/wd4996",
                    "/I..\core", "/DFENG_FP16_WEIGHTS=1", "/DFENG_MAX_SCORES=32768",
                    "/DFENG_USE_OMP=1", "/openmp")
        # 驱动走 C++23：/TP 强制按 C++ 编译；异常/RTTI 对纯 C 核心无意义，
        # 但 PC 驱动自己也遵守"零异常"约定，所以保留关闭开关
        $cxxFlags = @("/nologo", "/c", "/O2", "/std:c++latest", "/TP", "/utf-8",
                      "/EHs-c-", "/GR-", "/D_HAS_EXCEPTIONS=0", "/wd4996", "/I..\core",
                      "/DFENG_USE_OMP=1", "/openmp", "/DFENG_FP16_WEIGHTS=1",
                      "/DFENG_MAX_SCORES=32768")
        $kvQ2 = @("/DFENG_KV_Q2=1", "/DFENG_KV_Q2_BLOCK=8")

        Write-Host "[1/5] 编译核心（C23）q2 / fp32 / i8 三套 ..." -ForegroundColor Cyan
        if ((Invoke-Native "cl" ($cFlags + $kvQ2 + $core + "/Fo$objQ2\")) -ne 0) {
            throw "cl 编译失败（核心 q2）"
        }
        if ((Invoke-Native "cl" ($cFlags + $core + "/Fo$objFp32\")) -ne 0) {
            throw "cl 编译失败（核心 fp32）"
        }
        if ((Invoke-Native "cl" ($cFlags + @("/DFENG_KV_INT8=1") + $core + "/Fo$objI8\")) -ne 0) {
            throw "cl 编译失败（核心 i8）"
        }
        Write-Host "[2/5] 编译驱动（C++23）q2 组 ..." -ForegroundColor Cyan
        if ((Invoke-Native "cl" ($cxxFlags + $kvQ2 + $q2Drivers + "/Fo$objQ2\")) -ne 0) {
            throw "cl 编译失败（驱动 q2）"
        }
        Write-Host "[3/5] 编译 pc_check / pc_kv_suite_i8 驱动 ..." -ForegroundColor Cyan
        if ((Invoke-Native "cl" ($cxxFlags + @("pc_check.cpp") + "/Fo$objFp32\")) -ne 0) {
            throw "cl 编译失败（pc_check）"
        }
        if ((Invoke-Native "cl" ($cxxFlags + @("/DFENG_KV_INT8=1") + @("pc_kv_suite.cpp") +
                "/Fo$objI8\")) -ne 0) {
            throw "cl 编译失败（pc_kv_suite i8）"
        }

        Write-Host "[4/5] 链接 13 个产物 ..." -ForegroundColor Cyan
        function Get-Objs([string]$dir, [string[]]$sources) {
            $sources | ForEach-Object { Join-Path $dir ([IO.Path]::GetFileNameWithoutExtension($_) + ".obj") }
        }
        $q2Core = Get-Objs $objQ2 $core
        foreach ($d in $q2Drivers) {
            $name = [IO.Path]::GetFileNameWithoutExtension($d)
            $out = switch ($name) {
                "pc_chat"    { "pc_chat_q2b8.exe" }
                "pc_kv_suite" { "pc_kv_suite_q2b8.exe" }
                "pc_mt_suite" { "pc_mt_suite.exe" }
                default      { "$name.exe" }
            }
            if ((Invoke-Native "link" (@("/nologo", ("/OUT:" + (Join-Path $PSScriptRoot $out)),
                    (Join-Path $objQ2 ("{0}.obj" -f $name)) ) + $q2Core)) -ne 0) {
                throw "链接失败：$name"
            }
        }
        if ((Invoke-Native "link" (@("/nologo",
                ("/OUT:" + (Join-Path $PSScriptRoot "pc_check.exe")),
                (Join-Path $objFp32 "pc_check.obj")) + (Get-Objs $objFp32 $core))) -ne 0) {
            throw "链接失败：pc_check"
        }
        if ((Invoke-Native "link" (@("/nologo",
                ("/OUT:" + (Join-Path $PSScriptRoot "pc_kv_suite_i8.exe")),
                (Join-Path $objI8 "pc_kv_suite.obj")) + (Get-Objs $objI8 $core))) -ne 0) {
            throw "链接失败：pc_kv_suite_i8"
        }
        Write-Host "[5/5] 已生成 13 个产物（核心 C23 + 驱动 C++23）" -ForegroundColor Green
    } else {
        # ---------------- MinGW：gcc -std=c23 核心 + g++ -std=c++23 驱动 ----------------
        $gcc = $env:FENG_GCC; if (-not $gcc) { $gcc = "gcc" }
        $gxx = $env:FENG_GXX; if (-not $gxx) { $gxx = "g++" }
        $cFlags = @("-std=c23", "-O2", "-I..\core", "-DFENG_FP16_WEIGHTS=1",
                    "-DFENG_MAX_SCORES=32768", "-DFENG_USE_OMP=1", "-fopenmp")
        $cxxFlags = @("-std=c++23", "-fno-exceptions", "-fno-rtti", "-fno-threadsafe-statics",
                       "-DFENG_FP16_WEIGHTS=1", "-DFENG_MAX_SCORES=32768",
                      "-O2", "-DFENG_USE_OMP=1", "-fopenmp", "-I..\core")
        $objRoot = Join-Path $PSScriptRoot "mingw_obj"
        $objQ2 = Join-Path $objRoot "q2"
        $objFp32 = Join-Path $objRoot "fp32"
        $objI8 = Join-Path $objRoot "i8"
        New-Item -ItemType Directory -Force -Path $objQ2, $objFp32, $objI8 | Out-Null

        function Compile-Group([string]$objDir, [string[]]$defs) {
            foreach ($src in $core) {
                $o = Join-Path $objDir ([IO.Path]::GetFileNameWithoutExtension($src) + ".o")
                if ((Invoke-Native $gcc (@("-c") + $defs + $cFlags + @($src, "-o", $o))) -ne 0) {
                    throw "gcc 编译失败：$src"
                }
            }
        }
        Write-Host "[1/3] gcc -std=c23 编译核心三套 ..." -ForegroundColor Cyan
        Compile-Group $objQ2 @("-DFENG_KV_Q2=1", "-DFENG_KV_Q2_BLOCK=8")
        Compile-Group $objFp32 @()
        Compile-Group $objI8 @("-DFENG_KV_INT8=1")

        Write-Host "[2/3] g++ -std=c++23 编译并链接 13 个产物 ..." -ForegroundColor Cyan
        $q2Core = $core | ForEach-Object { Join-Path $objQ2 ([IO.Path]::GetFileNameWithoutExtension($_) + ".o") }
        $fp32Core = $core | ForEach-Object { Join-Path $objFp32 ([IO.Path]::GetFileNameWithoutExtension($_) + ".o") }
        $i8Core = $core | ForEach-Object { Join-Path $objI8 ([IO.Path]::GetFileNameWithoutExtension($_) + ".o") }
        $drv = Join-Path $objQ2 "drv"
        New-Item -ItemType Directory -Force -Path $drv | Out-Null
        foreach ($d in $q2Drivers) {
            $name = [IO.Path]::GetFileNameWithoutExtension($d)
            $out = switch ($name) {
                "pc_chat"    { "pc_chat_q2b8.exe" }
                "pc_kv_suite" { "pc_kv_suite_q2b8.exe" }
                "pc_mt_suite" { "pc_mt_suite.exe" }
                default      { "$name.exe" }
            }
            if ((Invoke-Native $gxx ($cxxFlags + @("-DFENG_KV_Q2=1", "-DFENG_KV_Q2_BLOCK=8") +
                    @($d, "-o", $out) + $q2Core + @("-lm"))) -ne 0) {
                throw "g++ 链接失败：$name"
            }
        }
        if ((Invoke-Native $gxx ($cxxFlags + @("pc_kv_suite.cpp", "-DFENG_KV_INT8=1",
                "-o", "pc_kv_suite_i8.exe") + $i8Core + @("-lm"))) -ne 0) {
            throw "g++ 链接失败：pc_kv_suite_i8"
        }
        if ((Invoke-Native $gxx ($cxxFlags + @("pc_check.cpp", "-o", "pc_check.exe") +
                $fp32Core + @("-lm"))) -ne 0) {
            throw "g++ 链接失败：pc_check"
        }
        Write-Host "[3/3] 已生成 13 个产物（核心 gcc -std=c23 + 驱动 g++ -std=c++23）" -ForegroundColor Green
    }

    Write-Host "跑一下 tool 单测：" -ForegroundColor Cyan
    # 必须用 $PSScriptRoot：第 25 行已经 Push-Location 到 pc\，
    # 再写相对路径 `pc\xxx.exe` 会变成 pc\pc\xxx.exe —— PowerShell 解析不成路径，
    # 转去加载名为 pc 的模块并报 "无法加载模块 'pc'"，构建成功也返回 exit 1。
    foreach ($t in "pc_calc_test", "pc_tools_test", "pc_mem_test", "pc_sample_test") {
        $rc = Invoke-Native (Join-Path $PSScriptRoot "$t.exe") @()
        if ($rc -ne 0) { throw "单测失败：$t（exit $rc）" }
    }
} finally {
    Pop-Location
}
