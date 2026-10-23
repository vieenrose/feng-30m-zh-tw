r"""在 MSYS/Git Bash（或任何 shell）里可靠地启动 ESP-IDF 的 `idf.py`。

为什么需要它：
1. `idf_tools.py` 见到环境变量 `MSYSTEM` 就直接退出（"MSys/Mingw is not
   supported"）。Git Bash 每次调用都会注入它，`unset` 也压不住（父进程重新导出）。
2. Windows 上还必须把 `IDF_TOOLS_PATH` 指到工具链实际安装位置，否则
   `idf.py` 去默认的 `C:\Users\<user>\.espressif` 找不到 python 环境。
3. Store 版 `pwsh` 存根会让 `-File` 之后的开关参数丢失，`flash.ps1` 因此
   在某些机器上根本跑不起来。

这个脚本先摘掉 MSYS 变量、补齐 IDF 路径，再以子进程执行 `idf.py`，绕开以上三条。

用法：
    python board/tools/idf_launch.py build
    python board/tools/idf_launch.py --port COM19 flash
    python board/tools/idf_launch.py -DFENG_NO_BONUS=ON build

环境变量（都有默认值，换机器时按需覆盖）：
    IDF_PATH         esp-idf 源码目录（默认 F:\esp\v5.5.5\esp-idf）
    IDF_TOOLS_PATH   工具链安装根（默认 F:\Espressif）
    IDF_PYTHON       该环境里的 python.exe（默认 $IDF_TOOLS_PATH/python_env/idf5.5_py3.12_env/Scripts/python.exe）
"""
import os
import subprocess
import sys

# 必须在任何子进程之前摘掉，否则 idf_tools.py 直接拒绝运行
for var in ("MSYSTEM", "MSYS", "MINGW_PREFIX", "MSYSCON"):
    os.environ.pop(var, None)

IDF = os.environ.get("IDF_PATH", r"F:\esp\v5.5.5\esp-idf")
TOOLS = os.environ.get("IDF_TOOLS_PATH", r"F:\Espressif")
PY = os.environ.get(
    "IDF_PYTHON",
    os.path.join(TOOLS, "python_env", "idf5.5_py3.12_env", "Scripts", "python.exe"),
)

if not os.path.isdir(IDF):
    sys.exit(f"找不到 IDF_PATH={IDF}（用环境变量指到 esp-idf 目录）")
if not os.path.isfile(PY):
    sys.exit(f"找不到 IDF_PYTHON={PY}（用环境变量指到该环境的 python.exe）")

os.environ["IDF_PATH"] = IDF
os.environ["IDF_TOOLS_PATH"] = TOOLS

# 在 board/ 目录下执行（idf.py 要求当前目录是工程根）
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(here)

cmd = [PY, os.path.join(IDF, "tools", "idf.py")] + sys.argv[1:]
sys.exit(subprocess.call(cmd))
