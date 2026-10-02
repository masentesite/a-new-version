#!/usr/bin/env bash
# ============================================================================
#  GroundingDINO 云端环境部署 —— uv 版
#
#  在【云端实例】里执行,进到项目目录跑这一条就够了:
#
#      cd <项目>/grounding_dino
#      bash setup_cloud_uv.sh
#
#  ────────────────────────────────────────────────────────────────────────────
#  为什么用 uv(与 setup_cloud_conda.sh 的区别)
#
#  1. 体积与时间(2026-09-30 实测,两条路都跑通过)
#         uv    : .venv 7.4G,无 solve 阶段,uv.lock 直接下
#         conda : env 12.6G + pkgs 18G,下载 4.3G,Solve 284 秒
#     根因:pip wheel 是自包含的;conda 把 torch 拆成 20 多个外部库
#     (libtorch 2159M / libmagma 1264M / libcudnn 1000M ...),全是 DT_NEEDED
#     级硬依赖,删不掉。而且 environment.lock.yml 只是版本清单,
#     `conda env create -f` 照样要重新 solve —— conda 没有"锁下载计划"的机制。
#
#  2. 版本保真
#     uv.lock 锁的是 torch 2.10.0+cu128(本机已验证过的那个版本)。
#     conda-forge 没有 cu128 变体,走 conda 必然被顶到 2.13.0+cu129,
#     跨 3 个 minor。论复现,uv 复现的是"锁文件里那个确切版本"。
#
#  3. 三个源全部有国内镜像(本脚本第 2 步会逐个探活)
#         PyPI 包      → pypi.tuna.tsinghua.edu.cn   (uv.lock 已改好)
#         torch cu128  → mirrors.nju.edu.cn/pytorch  (PEP503,sha256 与官方一致)
#         CPython 3.10 → mirrors.nju.edu.cn/github-release (GitHub 会 302 到
#                        release-assets.githubusercontent.com,国内常年不通)
#
#  ────────────────────────────────────────────────────────────────────────────
#  ⚠️ uv 路线唯一的真难点:nvcc(第 6 步)
#
#  upstream 的 GroundingDINO/setup.py 里 get_extensions() 有个【静默降级】分支:
#
#      if CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ):
#          ...  extension = CUDAExtension; sources += source_cuda
#      else:
#          print("Compiling without CUDA")
#          return None          # ← 不编译任何扩展,但安装照样"成功"
#
#  而没有 _C 时,ms_deform_attn.py:334 的分派条件是
#      if torch.cuda.is_available() and value.is_cuda:   # 看张量在不在 GPU,不是看 _C 导没导进来
#  所以在 4090 上【不会】走 multi_scale_deformable_attn_pytorch 那个纯 PyTorch 回退,
#  而是直接 NameError: name '_C' is not defined —— 而且失败点在第一次前向,
#  不在安装期。第 8 步因此必须验证 _C 真的编出来了,不能只看 import 通过。
#
#  版本比对规则(torch 2.10 实测,别再信"只特批 11.0/11.1"那套旧说法):
#      cpp_extension._check_cuda_version:
#          if cuda_ver != torch_cuda_version:
#              if cuda_ver.major != torch_cuda_version.major:
#                  raise RuntimeError(...)    # 只有 MAJOR 不同才炸
#              logger.warning(...)            # major 相同、minor 不同 → 只警告,照编
#  所以只要 nvcc 的 major 是 12 就能用;镜像自带的 13.0 会炸(13≠12)。
#
#  ⚠️ 但"major=12"只是【第一道】条件:有 nvcc ≠ 能用。还有一批头文件必须落在
#     $CUDA_HOME/include 下 —— cuda.h / cuda_runtime.h / cuda_runtime_api.h /
#     cublas_v2.h / cublasLt.h / cusparse.h / cusolverDn.h,外加 crt/host_defines.h
#     与 crt/host_config.h(后两个是前两个无条件 include 的,漏了同样炸),
#     再加 nv/target(CCCL / libcu++ 的,torch 头链上的 cuda_fp16.h:4492 无条件
#     `#include <nv/target>`,见下面第二次踩的记录)。
#     缺任何一个都在第 7 步死在第一个 .cpp 上,报 "cuda_runtime_api.h: No such
#     file or directory" —— 看着像 torch / uv / uv.lock 的问题,其实是 CUDA_HOME
#     是个半成品(只装了 nvcc,或上次装到一半)。2026-09-30 实测踩过这个坑。
#     第 6 步现在两头都查:候选必须 major 匹配【且】头文件齐全;不齐就地补
#     (venv 里 torch 自带的 nvidia-*-cu12 优先,零下载;crt/ 与 nv/ 在 pip 那套里
#      都没有 —— 前者取 redist 的 cuda_nvcc include/(约 78MB),
#      后者取 redist 的 cuda_cccl(整包才 1MB));补不齐直接 die 并给出
#     三条出路,不再把这颗雷留到第 7 步。
#
#  ⚠️ 2026-09-30 第二次实跑:补完那 9 个头,第 7 步换了个死法 ——
#     cuda_fp16.h:4492:10: fatal error: nv/target: No such file or directory
#     即上面说的 CCCL。它是【独立组件】cuda_cccl(redist 里 12.9.27),
#     cuda_cudart 的 include/ 里【没有】nv/。所以 nv/target 是第 10 个硬需求,
#     组件清单里也必须带上 cuda_cccl —— 否则"头文件齐了"的判据本身就是漏的。
#
#  ────────────────────────────────────────────────────────────────────────────
#  可选参数:
#      --check         只体检,不做任何改动
#      --jobs N        编译 _C 的并行数(默认 4)
#      --force         已存在的 .venv 也删掉重建
#      --cuda-home DIR 手动指定 CUDA_HOME(nvcc 的 major 必须等于 12)
#                      优先级高于自动扫描,用于镜像里的 12.x 装在冷门路径时
#      --no-auto-cuda  镜像里没有 major 匹配的 nvcc 时【不要】自动下载(默认会下)。
#                      默认行为:从 NVIDIA 官方 redist 拉一套 CUDA 12.x 到数据盘,
#                      只解需要的组件,不装驱动、不碰系统 CUDA。
#      --pip-nvcc      【已废弃】2026-09-30 实测:PyPI 的 nvidia-cuda-nvcc-cu12
#                      里没有 nvcc 可执行文件。用了会直接报错退出,见第 6 步。
#      --mirror-tuna URL / --mirror-pytorch URL / --mirror-python URL
#                      覆盖三个镜像地址
#      --foreground    不要自动进 screen(默认会自动进,见第 0 步)
#      SKIP_HDR_CHECK=1(环境变量,不是参数)跳过第 6 步的"自动补头 + 缺头即 die"。
#                      只在你的头文件是走 CPATH / 额外 -I 从别处喂进去时才需要。
# ============================================================================

# ⚠️ 刻意【不】加 -u(set -u)。
# 理由与 setup_cloud_conda.sh 相同,但这里还有一个更直接的原因:
# uv 会去跑上游 setup.py,而 setup.py 内部又会 subprocess 调 pip。
# 这条链上任何一个环节引用了未定义变量,-u 都会让【整个 shell】退出,
# 而不是只让那个子进程失败 —— 现场表现是"装到一半静默死亡"。
# pipefail 已经足够暴露真实错误码。
set -o pipefail

# ------------------------------ 版本锚点 ------------------------------
TORCH_CUDA_MAJOR="12"            # torch 是 cu128 → nvcc 的 major 必须是 12
MIN_UV_VERSION="0.12.10"         # pyproject.toml 的 build-system 要求 uv_build>=0.12.10
PY_VER="3.10"                    # 与 .python-version 一致
MIRROR_TUNA="${MIRROR_TUNA:-https://pypi.tuna.tsinghua.edu.cn/simple}"
MIRROR_PYTORCH="${MIRROR_PYTORCH:-https://mirrors.nju.edu.cn/pytorch/whl/cu128}"
MIRROR_PYTHON="${MIRROR_PYTHON:-https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone}"

# ------------------------------ 参数 ------------------------------
MODE="deploy"; FORCE=0; USE_PIP_NVCC=0; AUTO_CUDA=1; MAX_JOBS="${MAX_JOBS:-4}"; FOREGROUND=0
CUDA_HOME_ARG=""                 # --cuda-home 指定,优先级最高
ORIG_ARGS=("$@")

usage() { sed -n '2,/^# ===/p' "$0" | sed '$d' | sed 's/^# \?//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --check)          MODE="check" ;;
    --jobs)           MAX_JOBS="$2"; shift ;;
    --force)          FORCE=1 ;;
    --pip-nvcc)       USE_PIP_NVCC=1 ;;
    --no-auto-cuda)   AUTO_CUDA=0 ;;
    --cuda-home)      CUDA_HOME_ARG="$2"; shift ;;
    --mirror-tuna)    MIRROR_TUNA="$2"; shift ;;
    --mirror-pytorch) MIRROR_PYTORCH="$2"; shift ;;
    --mirror-python)  MIRROR_PYTHON="$2"; shift ;;
    --foreground)     FOREGROUND=1 ;;
    -h|--help)        usage ;;
    *) echo "未知参数:$1"; usage ;;
  esac
  shift
done

# ------------------------------ 输出 ------------------------------
C_R=$'\033[1;31m'; C_G=$'\033[1;32m'; C_Y=$'\033[1;33m'; C_C=$'\033[1;36m'; C_0=$'\033[0m'
say()  { printf '\n%s==> %s%s\n' "$C_C" "$*" "$C_0"; }
ok()   { printf '  %s✓%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '  %s!%s %s\n' "$C_Y" "$C_0" "$*"; }
die()  { printf '\n  %s✗ %s%s\n\n' "$C_R" "$*" "$C_0"; exit 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || { echo "无法进入 $PROJECT_DIR"; exit 1; }
LOG="$PROJECT_DIR/setup_uv.log"
SCREEN_NAME="gdino-setup-uv"
DATA_DISK="/root/autodl-tmp"

# ============================ 0. 防断线 ============================
# 这个脚本要下 ~3G(uv 缓存)+ 7.4G 落盘,还要编译 _C,合计 20 分钟以上。
# 2026-09-30 在 conda 那条路上实测过一次事故:脚本裸跑在 SSH 会话里,
# 会话一断 SIGHUP 广播给整个进程组,5 路并行下载在 0.3 秒内一起定格
# (6 个 .partial 的 mtime 落在 0.28 秒内),854M 白下。
# 同时停下 = 父进程没了;网络问题会让它们一个接一个地失败、时间戳是散的。
# 所以默认把自己塞进 screen。
if [ "$MODE" != "check" ] && [ "$FOREGROUND" != 1 ] && [ "${GDINO_IN_SCREEN:-0}" != 1 ]; then
  if [ -z "${STY:-}" ] && [ -z "${TMUX:-}" ]; then
    if command -v screen >/dev/null 2>&1; then
      _cmd="GDINO_IN_SCREEN=1 bash $(printf '%q' "$0")"
      for _a in "${ORIG_ARGS[@]}"; do _cmd="$_cmd $(printf '%q' "$_a")"; done
      screen -dmS "$SCREEN_NAME" bash -c "$_cmd > $(printf '%q' "$LOG") 2>&1"
      printf '\n%s 已在 screen 会话 [%s] 里启动%s\n' "$C_G" "$SCREEN_NAME" "$C_0"
      cat <<EOF

  这一步要跑 20 分钟以上,裸跑在 SSH 里的话,断线会把已下载的包一起带走。
  所以放进 screen 了,断线也不影响。

      看进度 : tail -f $LOG
      进会话 : screen -r $SCREEN_NAME
      退出会话: 先按 Ctrl-A 再按 D(不会中断脚本)

  不想用 screen(请自己挂 nohup):bash $0 --foreground

EOF
      exit 0
    else
      warn "没有 screen 也没有 tmux —— 请务必自己用 nohup 跑,否则断线会前功尽弃"
    fi
  fi
fi

# ============================ 1. 体检 ============================
say "1/8 环境体检"
printf '  主机 : %s\n' "$(hostname)"
printf '  系统 : %s\n' "$( . /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-未知}" )"
printf '  项目 : %s\n' "$PROJECT_DIR"
printf '  磁盘 :\n'; df -h / "$DATA_DISK" 2>/dev/null | sed 's/^/         /'

GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
[ -n "$GPU" ] || die "没有检测到 GPU —— 是不是开了「无卡模式」?编译 _C 必须有卡"
ok "GPU:$GPU"

# 算力直接从卡上读,别写死(换卡不用改脚本)
GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
# ⚠️ 两种写法别混:nvidia-smi 给的是 8.9(sm_89,len=3),cuobjdump 列的是
#    sm_89(去掉点)。第 8 步拿 sm_$GPU_CAP 去 grep 永远 grep 不到 ——
#    2026-09-30 实测在 4080 SUPER 上假报了一次"没看到 sm_8.9",
#    而 _C.so 里明明有 2 个 sm_89 cubin。这就是那个 bug 的修法。
GPU_ARCH="${GPU_CAP//./}"; GPU_ARCH="${GPU_ARCH:-89}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${GPU_CAP:-8.9}}"
export MAX_JOBS
ok "算力 $GPU_CAP → TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\"  MAX_JOBS=$MAX_JOBS"

[ -f "$PROJECT_DIR/uv.lock" ] || die "找不到 uv.lock —— 请在 grounding_dino/ 下执行"
[ -f "$PROJECT_DIR/GroundingDINO/setup.py" ] \
  || die "找不到 GroundingDINO/setup.py —— 子模块没 clone 全?"
ok "uv.lock 与 GroundingDINO/ 就位"

# 锁文件里若还有官方主机名,说明用的是没换过源的旧锁 —— 在国内会非常慢。
# ⚠️ 别写 `|| echo 0`:grep -c 无匹配时会【既打印 0 又返回非零】,
#    再 echo 一个 0 就得到 "0\n0",后面那个 != 0 的判断会白白成立。
_off="$(grep -cE 'pythonhosted|pypi\.org|download.*pytorch\.org' "$PROJECT_DIR/uv.lock" 2>/dev/null)"
_off="${_off:-0}"
if [ "$_off" != 0 ]; then
  warn "uv.lock 里还有 $_off 处官方源地址(这份锁没换过镜像),国内会明显变慢"
else
  ok "uv.lock 已全部指向国内镜像(官方主机名 0 残留)"
fi

# 上游 .so 是编进【源码树】的(editable + inplace build)。若上一台机器留下的
# .so 还在,而这次编译又被 setuptools 判定为"无需重编",就会静默用错架构的库。
# 本机的 .so 编进了 sm_35..sm_90 + sm_120,所以搬到 4090 上大概率还能跑 ——
# 正因为它"还能跑",错了也不会报错,所以这里显式提醒。
_so="$(find "$PROJECT_DIR/GroundingDINO" -maxdepth 3 -name '_C*.so' 2>/dev/null | head -1)"
if [ -n "$_so" ]; then
  warn "源码树里已有 _C.so:$(basename "$_so")"
  printf '      它是上一台机器编的。本脚本会重新编译覆盖;若想确保干净可先删:\n'
  printf '          rm -f %s\n' "$_so"
fi

case "$PROJECT_DIR" in
  /root/autodl-tmp/*|/root/autodl-fs/*) ok "项目在数据盘" ;;
  /root/*) warn "项目在系统盘(30G)。.venv 要占 7.4G,见第 3 步的兜底" ;;
esac

# ============================ 2. 镜像 ============================
say "2/8 镜像探活"
# 【为什么是 HEAD 而不是"真去解析一次"】
# conda 那条路的旧版用 `conda search` 验镜像,为了找一个包会把整个 channel 的
# repodata 拉下来解析,实测耗时 4 分钟,而屏幕上一行 `...working...` 看着像卡死。
# 一个 HEAD 请求就能验完 DNS + TLS + HTTP + 路径存在,实测 0.66 秒。
# 老实说这不省字节(装的时候反正要下同一份),只是让"镜像不通"在一秒内暴露。
probe_url() {  # $1 = 完整 URL → 打印 http code,可达则返回 0
  local url="$1" code=""
  code="$(curl -sI -o /dev/null -w '%{http_code}' --max-time 10 "$url" 2>/dev/null)"
  case "$code" in 200) printf '%s' "$code"; return 0 ;; esac
  # 有些镜像禁 HEAD,退化成只取 1 字节的 GET(206)
  code="$(curl -s -o /dev/null -r 0-0 -w '%{http_code}' --max-time 10 "$url" 2>/dev/null)"
  case "$code" in 200|206) printf '%s' "$code"; return 0 ;; esac
  printf '%s' "${code:-000}"; return 1
}

# 探针是只读的,所以 --check 下也照跑 —— 体检的意义正在于提前发现镜像不通。
_fail=0
check_one() {  # $1 = 标签, $2 = 完整 URL
  printf '  %-14s ' "$1"
  if _c="$(probe_url "$2")"; then
    printf '%s✓%s HTTP %s\n' "$C_G" "$C_0" "$_c"
  else
    printf '%s✗%s HTTP %s\n' "$C_R" "$C_0" "$_c"; _fail=1
  fi
}
# 注意三个 URL 的拼法各不相同:$MIRROR_TUNA 已经含 /simple,
# $MIRROR_PYTORCH 已经含 /whl/cu128,所以后面只接包名;
# python 那个探的是 base 本身(它的下一级是 <tag>/,tag 随时间变,不能写死)。
check_one "PyPI(tuna)"  "$MIRROR_TUNA/addict/"
check_one "torch(nju)"  "$MIRROR_PYTORCH/torch/"
check_one "python(nju)" "${MIRROR_PYTHON%/}/"
[ "$_fail" = 0 ] || die "有镜像不可达。换一个再试:
       · PyPI     : --mirror-tuna https://mirrors.aliyun.com/pypi/simple
       · torch    : --mirror-pytorch https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
       · CPython  : --mirror-python https://mirror.sjtu.edu.cn/github-release/astral-sh/python-build-standalone"
printf '  (注:python 那个路径形状是 <base>/<tag>/<文件名>,NJU 没有 /releases/download/ 段,\n'
printf '   正好对上 uv 拼 URL 的方式 —— 所以 UV_PYTHON_INSTALL_MIRROR 要填 base 本身)\n'

# ============================ 3. 目录规划 ============================
# 【为什么要把 uv 的东西也挪走】
# uv 缓存默认在 ~/.cache/uv:torch 那个 wheel 单个就 916M,全部下完约 3G。
# uv 托管的 CPython 默认在 ~/.local/share/uv/python(约 100M)。
# 系统盘只有 30G overlay,而 .venv 还要 7.4G —— 光靠"项目放数据盘"躲不掉缓存。
say "3/8 缓存目录"
export UV_CACHE_DIR="$DATA_DISK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$DATA_DISK/uv-python"

if [ ! -d "$DATA_DISK" ]; then
  warn "没有 $DATA_DISK,只能用系统盘 —— 注意 30G 要装 ~11G,会很紧"
  export UV_CACHE_DIR="$HOME/.cache/uv"
  export UV_PYTHON_INSTALL_DIR="$HOME/.local/share/uv/python"
fi

[ "$MODE" = "check" ] || mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" 2>/dev/null
ok "UV_CACHE_DIR        = $UV_CACHE_DIR"
ok "UV_PYTHON_INSTALL_DIR = $UV_PYTHON_INSTALL_DIR"

# .venv 按 uv 的默认规则落在项目目录下。项目在系统盘时,把 venv 也指到数据盘。
if [ "$FORCE" = 1 ] && [ "$MODE" != "check" ] && [ -d "$PROJECT_DIR/.venv" ]; then
  warn "--force:删除已有 .venv"
  rm -rf "$PROJECT_DIR/.venv"
fi

if [ ! -d "$DATA_DISK" ]; then :; elif [ "${PROJECT_DIR#/root/autodl-tmp/}" = "$PROJECT_DIR" ] \
   && [ "${PROJECT_DIR#/root/autodl-fs/}" = "$PROJECT_DIR" ]; then
  export UV_PROJECT_ENVIRONMENT="$DATA_DISK/gdino-venv"
  warn "项目在系统盘 → .venv 改到 $UV_PROJECT_ENVIRONMENT(否则 7.4G 压系统盘)"
fi

# ============================ 4. 装 uv ============================
say "4/8 安装 uv"
if command -v uv >/dev/null 2>&1; then
  ok "已有 uv $(uv --version 2>/dev/null | awk '{print $2}')"
else
  # uv 本身就在 PyPI 上,所以走 tuna 即可,不必碰 astral.sh / GitHub release。
  # (astral.sh 的安装脚本最终也是去拉 GitHub release,国内不一定通)
  command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 \
    || die "镜像里连 python 都没有,没法引导安装 uv"
  _py="$(command -v python3 || command -v python)"
  printf '  用 %s 从 tuna 装 uv ...\n' "$_py"
  if [ "$MODE" = "check" ]; then
    warn "(体检)未安装。实际执行会跑:$_py -m pip install uv -i $MIRROR_TUNA"
  else
    "$_py" -m pip install -q --upgrade uv -i "$MIRROR_TUNA" \
      || die "从 tuna 装 uv 失败。可换 --mirror-tuna https://mirrors.aliyun.com/pypi/simple"
    # pip 装完可能落在 ~/.local/bin,当前 shell 的 PATH 未必刷新
    command -v uv >/dev/null 2>&1 || export PATH="$HOME/.local/bin:$PATH"
    ok "uv 已安装:$(uv --version 2>/dev/null || echo '但不在 PATH')"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  # pyproject.toml 的 build-system 要 uv_build>=0.12.10;太老的 uv 会直接拒绝这个项目
  _uvv="$(uv --version 2>/dev/null | awk '{print $2}')"
  if [ "$(printf '%s\n%s\n' "$MIN_UV_VERSION" "$_uvv" | sort -V | head -1)" = "$MIN_UV_VERSION" ]; then
    ok "uv $_uvv ≥ $MIN_UV_VERSION"
  else
    die "uv $_uvv 太老,项目要求 ≥ $MIN_UV_VERSION(pyproject.toml 的 build-system)"
  fi
fi

# ============================ 5. 装 CPython ============================
say "5/8 准备 CPython $PY_VER"
# 【为什么这一节不能省】
# uv 会按 .python-version 下载【托管】的 CPython,默认源是 GitHub releases。
# 实测 GitHub 会 302 到 release-assets.githubusercontent.com —— 国内常年不通,
# 表现是 uv sync 卡在 "Downloading cpython-3.10.x" 不动。
# 改成 NJU 镜像(它镜像了 astral-sh/python-build-standalone 的全部 release)。
export UV_PYTHON_INSTALL_MIRROR="$MIRROR_PYTHON"
ok "UV_PYTHON_INSTALL_MIRROR=$UV_PYTHON_INSTALL_MIRROR"
printf '  (NJU 布局是 <base>/<tag>/<文件名>,实测 cpython-3.10.21 x86_64 linux 直出 42MB 无跳转)\n'

if [ "$MODE" = "check" ]; then
  warn "(体检)未安装 Python。实际执行会跑:uv python install $PY_VER"
else
  uv python install "$PY_VER" || die "安装 CPython $PY_VER 失败 —— 多半是镜像地址不对,用 --mirror-python 换"
  ok "CPython $PY_VER 就位:$(uv python find "$PY_VER" 2>/dev/null)"
fi

# ============================ 6. nvcc ============================
# uv 路线唯一的真难点。详见文件头 ⚠️:nvcc 的 major 必须等于 torch 的 CUDA major(12)。
say "6/8 定位 nvcc"

nvcc_major() {  # $1 = nvcc 路径 → 打印 major(如 12)
  "$1" --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1
}
nvcc_full() {   # $1 = nvcc 路径 → 打印 x.y
  "$1" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1
}

# ---------------- 自动下载 CUDA 12.x(NVIDIA 官方 redist) ----------------
# 扫不到 major 匹配的 nvcc 时,自己把一套装下来 —— 而不是打印一段话让人手敲。
#
# 不依赖 conda,只需要 curl + tar + python3:
#   <base>/redistrib_<ver>.json 是一份带 sha256 的清单,每个组件的
#   linux-x86_64.relative_path 就是 tarball 的路径。
#   tarball 解出单层 <组件>-linux-x86_64-<版本>-archive/{bin,include,lib,nvvm},
#   把这些目录的内容并到一起,就得到一个布局正确的 CUDA_HOME。
#
# ⚠️ 域名会 301 跳到 developer.download.nvidia.cn(国内 CDN),所以 curl 必须 -L。
#
# 组件清单是从 torch 的头文件倒推的,不是拍脑袋:
#   ATen/cuda/CUDAContextLight.h(CUDAContext.h 无条件 include 它)里
#   cuda_runtime_api.h / cusparse.h / cublas_v2.h / cublasLt.h / cusolverDn.h
#   都是无条件 include,而 csrc 三个源文件全 include 了 CUDAContext.h。
#
#   cuda_nvcc        bin/nvcc + bin/{ptxas,nvlink,fatbinary} + nvvm/bin/{cicc,cudafe++}
#   cuda_cudart      cuda.h / cuda_runtime.h / cuda_runtime_api.h + libcudart.so
#   cuda_cccl        nv/target(＋cuda/std/ cub/ thrust/ —— CCCL,header-only,整包 1MB)
#   libcublas        cublas_v2.h / cublasLt.h          ┐
#   libcusparse      cusparse.h                        ├ 只要 include/
#   libcusolver      cusolverDn.h                      ┘
#   cuda_cuobjdump   第 8 步校验 .so 里的算力用
#
#   ⚠️ cuda_cccl 是 2026-09-30 第二次实跑补上的:缺它报的是
#      "nv/target: No such file or directory"(不是 cudaXXX.h,所以第一次那版
#      清单里没有它 —— cuda.h/cuda_runtime.h 全都找得到,照样编不过)。
#      它只有 1MB,不值得为它做 headers-only 的特殊处理。
#
# ⚠️ 那三个库的 tarball 各含一个多架构的大 .so(合计 1.6G),但 readelf -d 实测
#    _C.so 的 NEEDED 里只有 libcudart —— cublas/cusparse/cusolver 根本没被链接,
#    编 _C 只需要它们的头文件。所以解包时只取 include/,省下约 1.6G 磁盘。
CUDA_REDIST_VER="${CUDA_REDIST_VER:-12.9.1}"
CUDA_REDIST_BASE="${CUDA_REDIST_BASE:-https://developer.download.nvidia.com/compute/cuda/redist}"
CUDA_REDIST_HEADERS_ONLY="${CUDA_REDIST_HEADERS_ONLY:-libcublas libcusparse libcusolver}"
CUDA_REDIST_COMPONENTS="${CUDA_REDIST_COMPONENTS:-cuda_nvcc cuda_cudart cuda_cccl libcublas libcusparse libcusolver cuda_cuobjdump}"

install_cuda_redist() {   # $1 = 目标目录(即未来的 CUDA_HOME)
  local dest="$1" ver="$CUDA_REDIST_VER" base="$CUDA_REDIST_BASE"
  local cache="$dest/.download" manifest list name rel sha size tgz got total avail_kb need_kb

  command -v python3 >/dev/null 2>&1 || { warn "需要 python3 解析清单,镜像里没有"; return 1; }
  command -v curl    >/dev/null 2>&1 || { warn "需要 curl,镜像里没有";                return 1; }
  mkdir -p "$cache" || return 1
  manifest="$cache/redistrib_$ver.json"; list="$cache/list.tsv"

  if [ ! -s "$manifest" ]; then
    curl -fsSL --retry 3 --connect-timeout 20 -o "$manifest" "$base/redistrib_$ver.json" \
      || { warn "取清单失败:$base/redistrib_$ver.json"; return 1; }
  fi
  ok "清单 redistrib_$ver.json($(wc -c <"$manifest" | tr -d ' ') 字节)"

  python3 - "$manifest" "$CUDA_REDIST_COMPONENTS" >"$list" <<'PY' || { warn "解析清单失败"; return 1; }
import json, sys
COMPONENTS = sys.argv[2].split()
m = json.load(open(sys.argv[1]))
for name in COMPONENTS:
    f = (m.get(name) or {}).get("linux-x86_64") or {}
    if not f.get("relative_path"):
        sys.stderr.write("清单里没有 %s 的 linux-x86_64\n" % name)
        sys.exit(1)
    print("%s\t%s\t%s\t%s" % (name, f["relative_path"],
                              f.get("sha256", ""), f.get("size", 0)))
PY

  total=0
  while IFS=$'\t' read -r name rel sha size; do total=$((total + ${size:-0})); done <"$list"
  avail_kb="$(df -Pk "$dest" 2>/dev/null | awk 'NR==2{print $4}')"
  need_kb=$(( total / 1024 + 3 * 1024 * 1024 ))          # 下载量 + 3G 解包余量
  if [ -n "$avail_kb" ] && [ "$avail_kb" -lt "$need_kb" ]; then
    warn "可用 $((avail_kb / 1024))MB,本次约需 $((need_kb / 1024))MB —— 可能不够,先清点空间"
  fi

  while IFS=$'\t' read -r name rel sha size; do
    [ -n "${rel:-}" ] || continue
    tgz="$cache/$(basename "$rel")"
    if [ -s "$tgz" ]; then
      printf '  复用 %-16s %5s MB(上次已下)\n' "$name" "$((size / 1048576))"
    else
      printf '  下载 %-16s %5s MB  %s\n' "$name" "$((size / 1048576))" "$(basename "$rel")"
      curl -fL --retry 3 -C - --connect-timeout 20 -o "$tgz" "$base/$rel" \
        || { warn "$name 下载失败(重跑本脚本可断点续传)"; return 1; }
    fi
    if [ -n "$sha" ]; then
      got="$(sha256sum "$tgz" | cut -d' ' -f1)"
      [ "$got" = "$sha" ] || { warn "$name sha256 不符,已删除,重跑会重下:$tgz"; rm -f "$tgz"; return 1; }
    fi

    local _x="$cache/_x"; mkdir -p "$_x"
    case " $CUDA_REDIST_HEADERS_ONLY " in
      *" $name "*)
        tar -xJf "$tgz" -C "$_x" --wildcards '*/include/*' 2>/dev/null \
          || { warn "$name 解包(仅 include/)失败"; return 1; } ;;
      *)
        tar -xJf "$tgz" -C "$_x" || { warn "$name 解包失败"; return 1; } ;;
    esac
    for d in "$_x"/*/; do [ -d "$d" ] && cp -a "$d." "$dest/"; done
    rm -rf "$_x"
    ok "$name 就位"
  done <"$list"

  # ⚠️ cpp_extension 找的是 $CUDA_HOME/lib64 —— library_paths('cuda') 实测返回
  #    $CUDA_HOME/lib64 和 lib 两个,而 redist 归档里只有 lib/。补个软链,
  #    否则链接期 -lcudart 找不到(症状是 undefined reference to cudaXxx)。
  [ -d "$dest/lib" ] && [ ! -e "$dest/lib64" ] && ln -s lib "$dest/lib64"

  rm -rf "$cache"
  return 0
}

# ---------------- CUDA 头文件体检 + 就地补头 ----------------
# ⚠️ 判据不是"nvcc --version 能跑",而是【编 _C 要的那几个头在不在】。
#    2026-09-30 实测踩过:只有 nvcc、没有 include/ 的"最小工具链"照样让 uv sync
#    在第一个 .cpp 上炸,报的是 "cuda_runtime_api.h: No such file or directory",
#    看着像 torch 或 uv 的问题,其实是 CUDA_HOME 缺头。
#
#    依赖链(torch 2.10 实测,两条都是【无条件】include,不在 #if 里):
#      csrc/*.cpp|*.cu → ATen/cuda/CUDAContext.h → CUDAContextLight.h
#        → <cuda_runtime_api.h> → cuda_runtime_api.h:148  crt/host_defines.h
#        → <cusparse.h> <cublas_v2.h> <cublasLt.h> <cusolverDn.h>
#      而 cuda_runtime.h:82 无条件 include crt/host_config.h
#    所以 include/crt/ 和那 7 个一样是硬需求 —— 少了它报错文件名是
#    crt/host_defines.h,很容易被误判成"nvcc 版本不对"去查半天。
#
#    第二次实跑又补了一个:nv/target(CCCL / libcu++)。cuda_fp16.h:4492 在
#    `#if !defined(__CUDACC_RTC__)` 下【无条件】include 它,而 torch 那串头
#    (ATen/cuda/CUDAContext.h → … → cuda_fp16.h)必然走到。它在 redist 的
#    【独立组件 cuda_cccl】里,cuda_cudart 的 include/ 下没有 nv/ —— 所以
#    "查了这 9 个头都齐"并不等于能编,这就是那版的漏洞。
#    ⚠️ 别把它和 crt/ 混一起:来源不同(crt→cuda_nvcc,nv→cuda_cccl),
#       补的地方也就不同,分开一个变量放。
CUDA_HDRS="cuda.h cuda_runtime.h cuda_runtime_api.h cublas_v2.h cublasLt.h cusparse.h cusolverDn.h"
CUDA_HDRS_CRT="crt/host_defines.h crt/host_config.h"
CUDA_HDRS_CCCL="nv/target"

cuda_headers_ok() {   # $1 = CUDA_HOME;0 = 齐全
  local d="${1%/}" h
  for h in $CUDA_HDRS $CUDA_HDRS_CRT $CUDA_HDRS_CCCL; do [ -f "$d/include/$h" ] || return 1; done
  return 0
}
cuda_headers_missing() {   # $1 = CUDA_HOME → 打印缺哪些(空=齐全)
  local d="${1%/}" h
  for h in $CUDA_HDRS $CUDA_HDRS_CRT $CUDA_HDRS_CCCL; do
    [ -f "$d/include/$h" ] || printf '%s ' "$h"
  done
}
cuda_headers_report() {    # $1 = CUDA_HOME;逐项打勾,缺的顺带说补哪个 conda 包
  local d="${1%/}" h pkg
  for h in $CUDA_HDRS $CUDA_HDRS_CRT $CUDA_HDRS_CCCL; do
    if [ -f "$d/include/$h" ]; then ok "  include/$h"
    else
      case "$h" in
        crt/*|cuda.h|cuda_runtime.h|cuda_runtime_api.h) pkg="cuda-cudart-dev" ;;
        cublas_v2.h|cublasLt.h)                         pkg="libcublas-dev" ;;
        cusparse.h)                                     pkg="libcusparse-dev" ;;
        cusolverDn.h)                                   pkg="libcusolver-dev" ;;
        nv/*)                                           pkg="cuda-cccl" ;;
        *)                                              pkg="?" ;;
      esac
      warn "  缺 include/$h —— 编译过不去。补装 conda 包:$pkg"
    fi
  done
}

# 补头:先零下载,再只取缺的那一点 redist 内容(不重下 1.4G)。
#   ① venv —— torch wheel 依赖的 nvidia-*-cu12 pip 包里就带着这些头(实测):
#        nvidia/cuda_runtime/include/{cuda.h,cuda_runtime.h,cuda_runtime_api.h}
#        nvidia/cublas/include/{cublas_v2.h,cublasLt.h}
#        nvidia/cusparse/include/cusparse.h   nvidia/cusolver/include/cusolverDn.h
#      ⚠️ 光"venv 里有"没用:include_paths('cuda') 只返回 $CUDA_HOME/include,
#         【不看】pip 那个目录,所以必须复制过去。
#   ② crt/ —— pip 那套里【没有】(cudart 包里连 crt/ 目录都没有,而它的
#      cuda_runtime.h 却 include "crt/host_config.h",所以 pip 世界靠
#      nvidia-cuda-nvcc-cu12 补这一块)。这里改用 redist 的 cuda_nvcc,只解
#      include/(77MB),不碰已就位的 nvcc 本体,也不往 venv 里塞包
#      (塞了也会被后面 uv sync --frozen 清掉)。
#   ②' nv/ —— 同上,venv 那十几个 nvidia-*-cu12 包里【也没有】nv/target
#      (逐个 include/ 目录翻过)。所以它必然要联网取,好在 cuda_cccl 整包才 1MB。
#   ③ 万一 venv 里连 nvidia-cuda-runtime-cu12 都没有(全新机器,还没 uv sync),
#      再从 redist 取 cuda_cudart(1.4MB)。
# 返回 0 = 试过了(不代表补齐;调用方一律用 cuda_headers_ok 复核)。
topup_cuda_headers() {
  local home="${1%/}" sp d n=0 need="" _sc _sh rc=0
  mkdir -p "$home/include" || return 1
  cuda_headers_ok "$home" && return 0

  for sp in "${UV_PROJECT_ENVIRONMENT:-$PROJECT_DIR/.venv}"/lib/python*/site-packages/nvidia; do
    [ -d "$sp" ] || continue
    for d in "$sp"/*/include; do
      [ -d "$d" ] || continue
      cp -an "$d"/. "$home/include/" 2>/dev/null && n=$((n + 1))
    done
  done
  [ "$n" -gt 0 ] && ok "  从 venv 的 nvidia-*-cu12 并了 $n 个包的 include/(零下载)"

  cuda_headers_ok "$home" && return 0
  for h in $CUDA_HDRS_CRT; do
    [ -f "$home/include/$h" ] || { need="cuda_nvcc"; break; }
  done
  for h in $CUDA_HDRS_CCCL; do
    [ -f "$home/include/$h" ] || { need="$need cuda_cccl"; break; }
  done
  if [ ! -f "$home/include/cuda.h" ] || [ ! -f "$home/include/cuda_runtime.h" ] \
     || [ ! -f "$home/include/cuda_runtime_api.h" ]; then
    need="$need cuda_cudart"
  fi
  need="${need# }"
  [ -n "$need" ] || return 0

  warn "  仍缺:$(cuda_headers_missing "$home")  → 从 redist 只取 include/:$need"
  _sc="$CUDA_REDIST_COMPONENTS"; _sh="$CUDA_REDIST_HEADERS_ONLY"
  CUDA_REDIST_COMPONENTS="$need"
  CUDA_REDIST_HEADERS_ONLY="$need"
  install_cuda_redist "$home" || rc=1
  CUDA_REDIST_COMPONENTS="$_sc"; CUDA_REDIST_HEADERS_ONLY="$_sh"
  return $rc
}

# 补头总入口:零下载 → 还不行就整组件装(幂等,已下的 tarball 会复用)。
ensure_cuda_headers() {   # $1 = CUDA_HOME;0 = 齐了
  local home="${1%/}"
  cuda_headers_ok "$home" && return 0
  topup_cuda_headers "$home"
  cuda_headers_ok "$home" && return 0
  warn "  零下载补不齐(venv 里没有那套 pip 头),退回整组件安装(约 1.4G)"
  install_cuda_redist "$home" && cuda_headers_ok "$home"
}

CUDA_ROOT=""
# 逐个候选看过去,取第一个 major 匹配的。
# ⚠️ 别硬编码版本号。最初这里写死了 /usr/local/cuda-12.8 / -12.9 / -12.6 / -12.4,
#    镜像里若是别的 12.x(例如 -12.1 / -12.5)就整个扫不到,还会误报
#    "没有可用的 nvcc",把人骗去装一套根本不需要装的 toolkit。
#    改成 glob,并覆盖 conda env(conda 装的 cuda-nvcc 就落在 env 根下)。
_cands=()
[ -n "${CUDA_HOME_ARG:-}" ] && _cands+=("$CUDA_HOME_ARG")
[ -n "${CUDA_HOME:-}" ]     && _cands+=("$CUDA_HOME")
for _g in /usr/local/cuda* /opt/cuda* "${CONDA_PREFIX:-/nonexistent}" \
          "$HOME"/miniconda3/envs/* "$HOME"/anaconda3/envs/* \
          /root/miniconda3/envs/* /root/autodl-tmp/conda/envs/*; do
  [ -d "$_g" ] && _cands+=("$_g")
done

_seen_list=""
_repair_cand=""     # major 对、但头文件不全的"半成品":优先就地补,省一次 1.4G 下载
for _cand in "${_cands[@]}"; do
  [ -x "$_cand/bin/nvcc" ] || continue
  case "$_seen_list" in *"|$_cand|"*) continue ;; esac     # 去重
  _seen_list="$_seen_list|$_cand|"
  _m="$(nvcc_major "$_cand/bin/nvcc")"
  if [ "$_m" != "$TORCH_CUDA_MAJOR" ]; then
    printf '  候选 %-36s nvcc %s(major≠%s,跳过)\n' \
      "$_cand" "$(nvcc_full "$_cand/bin/nvcc")" "$TORCH_CUDA_MAJOR"
    continue
  fi
  # ⚠️ 有 nvcc ≠ 能用。major 对了还要看头文件齐不齐 —— 镜像里那种"只装了 nvcc
  #    的最小工具链"以前会被直接选中,然后在第 7 步炸在第一个 .cpp 上。
  if cuda_headers_ok "$_cand"; then
    printf '  候选 %-36s nvcc %s\n' "$_cand" "$(nvcc_full "$_cand/bin/nvcc")"
    CUDA_ROOT="$_cand"; break
  fi
  printf '  候选 %-36s nvcc %s ← 缺头文件:%s\n' \
    "$_cand" "$(nvcc_full "$_cand/bin/nvcc")" "$(cuda_headers_missing "$_cand")"
  [ -n "$_repair_cand" ] || _repair_cand="$_cand"
done

# 找到的只有半成品 → 就地补头(零下载优先),而不是为此再下一整套 1.4G。
if [ -z "$CUDA_ROOT" ] && [ -n "$_repair_cand" ]; then
  warn "$_repair_cand:nvcc major 对,但头文件不全(上次装到一半?只装了 nvcc?)"
  if [ "$MODE" = "check" ]; then
    warn "  (体检)未改动。实际执行会就地补头:venv 里 torch 自带的 nvidia-*-cu12"
    warn "  已含那 7 个中的大部分,通常只需再从 redist 取 crt/(约 78MB)"
    warn "  和 nv/target 所在的 cuda_cccl(约 1MB),不会重下 1.4G。"
    warn "  ⇒ 这种状态【不需要】手工 conda 装那套 4-5G 的工具链,直接跑本脚本就行。"
    say "体检模式(--check):到此为止,未做任何改动"
    exit 0
  elif ensure_cuda_headers "$_repair_cand"; then
    CUDA_ROOT="$_repair_cand"
    ok "补齐后可用:CUDA_HOME=$CUDA_ROOT"
  else
    warn "补不齐,继续按'镜像里没有可用的 nvcc'处理"
  fi
fi

# 扫不到 major 匹配的 → 自己装一套。把看到的版本说清楚,别让人以为"没装 CUDA"。
if [ -z "$CUDA_ROOT" ]; then
  _seen="$(for c in /usr/local/cuda*; do [ -x "$c/bin/nvcc" ] && printf '%s ' "$(nvcc_full "$c/bin/nvcc")"; done)"
  if [ -n "$_repair_cand" ]; then
    # 别在这里说"没有 major=12 的 nvcc" —— 上面刚打过候选行,自相矛盾。
    warn "nvcc major 对但不完整的那套($(nvcc_full "$_repair_cand/bin/nvcc"))没被采用"
  else
    warn "没有找到 major=$TORCH_CUDA_MAJOR 的 nvcc。镜像里实际有:${_seen:-无}"
  fi

  AUTO_CUDA_DIR="${AUTO_CUDA_DIR:-$DATA_DISK/cuda$TORCH_CUDA_MAJOR}"
  if ! mkdir -p "$AUTO_CUDA_DIR" 2>/dev/null; then
    AUTO_CUDA_DIR="$PROJECT_DIR/.cuda$TORCH_CUDA_MAJOR"
    mkdir -p "$AUTO_CUDA_DIR" || die "建不出 CUDA 安装目录"
    warn "数据盘不可写,改装到 $AUTO_CUDA_DIR"
  fi

  # ⚠️ 这里原来只查 bin/nvcc —— 于是"上次装到一半"的目录被当成装好的,每跑一次
  #    都在第 7 步撞同一堵墙(2026-09-30 实测:/root/autodl-tmp/cuda12 有 nvcc 12.9、
  #    没有 include/,报的是 cuda_runtime_api.h: No such file or directory)。
  #    改为连头文件一起复核;不全就地补,不整目录重下。
  if [ -x "$AUTO_CUDA_DIR/bin/nvcc" ] && cuda_headers_ok "$AUTO_CUDA_DIR"; then
    ok "已有装好的:$AUTO_CUDA_DIR (nvcc $(nvcc_full "$AUTO_CUDA_DIR/bin/nvcc"))"
    CUDA_ROOT="$AUTO_CUDA_DIR"
  elif [ -x "$AUTO_CUDA_DIR/bin/nvcc" ]; then
    warn "已有 nvcc 但头文件不全(半成品):$AUTO_CUDA_DIR"
    warn "  缺:$(cuda_headers_missing "$AUTO_CUDA_DIR")"
    if [ "$MODE" = "check" ]; then
      warn "  (体检)未改动。实际执行会就地补:零下载优先,不够才从 redist 取"
    elif ensure_cuda_headers "$AUTO_CUDA_DIR"; then
      CUDA_ROOT="$AUTO_CUDA_DIR"
      ok "补齐后可用:CUDA_HOME=$CUDA_ROOT"
    else
      warn "补不齐,当作没装过"
    fi
  elif [ "$MODE" = "check" ]; then
    warn "(体检)未下载。实际执行会从 NVIDIA redist 拉 CUDA $CUDA_REDIST_VER 到"
    warn "  $AUTO_CUDA_DIR(约 1.7G 下载,只解需要的组件,不碰系统 CUDA、不装驱动)"
  elif [ "$AUTO_CUDA" != 1 ]; then
    warn "--no-auto-cuda:跳过自动下载"
  elif [ "$USE_PIP_NVCC" = 1 ]; then
    :                                   # 交给下面的 --pip-nvcc 报错分支
  else
    if install_cuda_redist "$AUTO_CUDA_DIR" && cuda_headers_ok "$AUTO_CUDA_DIR"; then
      CUDA_ROOT="$AUTO_CUDA_DIR"
      ok "自动装好了:CUDA_HOME=$AUTO_CUDA_DIR"
    elif [ -x "$AUTO_CUDA_DIR/bin/nvcc" ]; then
      warn "装完了但头文件仍不全:$(cuda_headers_missing "$AUTO_CUDA_DIR")"
      warn "  这不该发生 —— 把上面几行连同 'ls $AUTO_CUDA_DIR/include' 一起发出来。"
    else
      warn "自动下载没成功。可重跑本脚本续传(已下的 tarball 会复用);"
      warn "或者照下面换成 conda 方案。"
    fi
  fi
fi

if [ -n "$CUDA_ROOT" ]; then
  export CUDA_HOME="$CUDA_ROOT"
  export PATH="$CUDA_HOME/bin:$PATH"
  ok "CUDA_HOME=$CUDA_HOME (nvcc $(nvcc_full "$CUDA_HOME/bin/nvcc"),major 匹配 torch 的 cu$TORCH_CUDA_MAJOR)"

  # 编 _C 要的不只是 nvcc,还有那 8 个头。依据与依赖链见上面 cuda_headers_ok 的
  # 注释(cuda_runtime_api.h:148 无条件 include crt/host_defines.h;cuda_runtime.h:82
  # 无条件 include crt/host_config.h;而 csrc 三个源文件全都 include 了
  # <ATen/cuda/CUDAContext.h>)。缺了就在这里补齐 / 说清楚 ——
  # 以前只 warn 一句就往下走,结果是第 7 步一屏编译器报错,还得回头猜。
  if ! cuda_headers_ok "$CUDA_HOME"; then
    warn "头文件不全,缺:$(cuda_headers_missing "$CUDA_HOME")"
    if [ "$MODE" = "check" ]; then
      :                       # 体检模式不改任何东西,下面照常逐项报告
    elif [ "${SKIP_HDR_CHECK:-0}" = 1 ]; then
      warn "SKIP_HDR_CHECK=1:跳过自动补头(头文件由你自己从别处喂时用)"
    else
      warn "就地补:venv 里 torch 自带的 nvidia-*-cu12 优先(零下载),不够才从 redist 取"
      ensure_cuda_headers "$CUDA_HOME" || true
    fi
  fi
  cuda_headers_report "$CUDA_HOME"
  if [ "$MODE" != "check" ] && [ "${SKIP_HDR_CHECK:-0}" != 1 ] && ! cuda_headers_ok "$CUDA_HOME"; then
    die "头文件仍不全,再往下跑必然在第 7 步炸在第一个 .cpp 上(报 cuda_runtime_api.h 或 nv/target 那类)。三条出路任选其一:
         · 往同一个 CUDA_HOME 补 conda 包(⚠️ 用 -p 不用 -n,别占系统盘):
             conda install -p $CUDA_HOME -c nvidia \\
               cuda-cudart-dev libcublas-dev libcusparse-dev libcusolver-dev cuda-cccl
         · --cuda-home 指向另一套【dev 齐全】的 CUDA 12.x
         · 让本脚本整装一套:unset CUDA_HOME 后删掉这个目录再重跑,
           会从 NVIDIA redist 拉全 7 个组件(约 1.4G)"
  fi
else
  if [ "$USE_PIP_NVCC" = 1 ]; then
    # 2026-09-30 实测判死刑,原实现(拼假 CUDA_HOME)已整段删除。
    # 留这个分支只为把证据说清楚,不再做任何尝试。
    die "--pip-nvcc 已废弃:PyPI 的 nvidia-cuda-nvcc-cu12 里【根本没有 nvcc】。
       实测装了 95MB,查它的 RECORD 只有 30 个文件,可执行的仅此一个:
         nvidia/cuda_nvcc/bin/ptxas
       外加 nvidia/cuda_nvcc/include/crt/*.h。
       nvcc 是个驱动器,要调 cudafe++ / cicc / ptxas / nvlink / fatbinary 一整套,
       这些包里一个都没有 —— 拼不出能用的 CUDA_HOME。
       → 请按下面的 conda 方案装一套真工具链。"
  else
    cat <<EOF

      需要一套 major=$TORCH_CUDA_MAJOR 的 CUDA 工具链:nvcc + 下面这批头文件。

      ── 装法(清华 conda-forge;依赖清单 2026-09-30 按 torch 2.10 头文件核实过)──

        # ⚠️ 用 -p 不用 -n:默认的 -n 会落到 /root/miniconda3/envs(系统盘 30G),
        #    全量工具链 4-5G 会把它撑爆。数据盘才是 .venv 该待的地方。
        conda create -y -p $DATA_DISK/cuda12 --override-channels \\
          -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \\
          cuda-version=12.9 \\
          cuda-nvcc cuda-cudart-dev \\
          libcublas-dev libcusparse-dev libcusolver-dev \\
          cuda-cuobjdump

        export CUDA_HOME=$DATA_DISK/cuda12
        export PATH="\$CUDA_HOME/bin:\$PATH"
        nvcc --version

      ⚠️ 写法要点:只把 cuda-version=12.9 当【锚】,其余六个【一个都不锁版本】。
         这是已验证的写法;给每个包都写 =12.9 会把 conda 逼进无解或退到老版本。

      为什么是这七个包 —— 不是"随便装个 nvcc 就行":

        cuda-nvcc         编译器本体。nvcc 是个驱动器,要调 cudafe++ / cicc /
                          ptxas / nvlink / fatbinary 一整套,单个可执行文件没用。
        cuda-cudart-dev   cuda.h / cuda_runtime.h / cuda_runtime_api.h
        libcublas-dev     cublas_v2.h / cublasLt.h
        libcusparse-dev   cusparse.h
        libcusolver-dev   cusolverDn.h
        cuda-cuobjdump    第 8 步验证 .so 里的 sm_$GPU_ARCH 要用它。它是【单独的包】,
                          不跟着 cuda-nvcc 走,漏了第 8 步只能跳过校验。

      前五个缺一不可。原因是 torch 的 ATen/cuda/CUDAContextLight.h 里这几行
      【无条件】include,而 csrc 三个源文件全都 include 了 CUDAContext.h:

          #include <cuda_runtime_api.h>
          #include <cusparse.h>
          #include <cublas_v2.h>
          #include <cublasLt.h>
          #ifdef CUDART_VERSION
          #include <cusolverDn.h>     ← CUDART_VERSION 上面刚定义过,等同无条件
          #endif

      💡 torch wheel 自带的 nvidia-*-cu12 pip 包里【已经含】这批头里的大部分
         (site-packages/nvidia/*/include)。实测 include_paths('cuda') 只返回
         torch/include + torch/include/torch/csrc/api/include + \$CUDA_HOME/include,
         【不含】pip 那个目录 —— 但【复制过去就能用】,本脚本第 6 步已经会自动做
         (零下载);pip 那套里没有 crt/ 和 nv/(nv/target 是 cuda_fp16.h 无条件
         include 的 CCCL 头,缺了报的是 "nv/target: No such file or directory"),
         这两个才需要从 redist 取 —— cuda_nvcc 的 include/(约 78MB)与
         cuda_cccl 的 include/(约 1MB),都不重下 1.4G。
         真要自己喂头文件、不想让它自动补:export SKIP_HDR_CHECK=1。

      12.9 与 torch 的 cu128 同为 major $TORCH_CUDA_MAJOR,按 torch 的规则只警告不报错。

      装完在【同一个 shell】里 export 那两行,然后重跑本脚本;
      或直接指定:bash $(basename "$0") --cuda-home $DATA_DISK/cuda12

      ── 其它出路 ──

      · 换一个自带 /usr/local/cuda-12.x 的镜像(交接文档实测实例 1 上有)。
        ⚠️ 但那个 12.x 必须是【dev 齐全】的 —— 只有 nvcc、没有上面那批头文件的
           "最小工具链"照样编不过,别看到 nvcc --version 能跑就以为成了。
      · 改用 setup_cloud_conda.sh —— nvcc 与 torch 由同一个 solver 解出,必然同源。
        代价是 12.6G env + 4.3G 下载 + 284 秒 solve,以及 torch 被顶到 2.13.0+cu129。

      先不改任何东西、只看现状:bash $(basename "$0") --check

EOF
    die "没有可用的 nvcc"
  fi
fi
export MAX_JOBS

# ============================ 7. uv sync ============================
say "7/8 uv sync --frozen"
# --frozen:严格按 uv.lock 来,一个字节都不重新解析。锁里是完整 URL + sha256,
#          所以这一步同时还是一次完整性校验 —— 镜像若被篡改会直接失败。
# no-build-isolation-package:已写进 pyproject.toml 的 [tool.uv],这里再显式带一次,
#          是为了防止有人用没更新过的旧 pyproject 跑这个脚本。
#          上游 setup.py 顶层会 import torch,隔离构建环境里没有 torch,它会自己去
#          pip install torch 然后失败(实测:uv lock --refresh 就是这么炸的)。
if [ "$MODE" = "check" ]; then
  warn "(体检)未执行。实际执行会跑:uv sync --frozen --no-build-isolation-package groundingdino"
  say "体检模式(--check):到此为止,未做任何改动"
  echo "  uv      : $(uv --version 2>/dev/null || echo 未装)"
  echo "  nvcc    : ${CUDA_HOME:-未定位}"
  echo "  下一步  : bash $(basename "$0")"
  exit 0
fi

warn "接下来会长时间静默,不是卡死:"
printf '       · 下载 ~3G(torch 单个 wheel 就 916M)\n'
printf '       · 编译 _C(唯一真正耗 CPU 的一步,MAX_JOBS=%s)\n' "$MAX_JOBS"
printf '     想确认它还活着:另开终端看网卡/CPU,别 Ctrl-C。\n'

uv sync --frozen --no-build-isolation-package groundingdino
RC=$?
[ $RC -eq 0 ] || die "uv sync 失败(退出码 $RC)。常见原因:
       · 校验失败(sha256 不符) → 镜像内容与官方不一致,换 --mirror-* 重试
       · No solution / lock 过期  → 说明 uv.lock 与 pyproject.toml 不同步,
         在【本机】跑 `uv lock` 后重新上传锁文件
       · 编译报错              → 看第 6 步的 nvcc 与头文件;major 必须等于 $TORCH_CUDA_MAJOR"

VENV_PY="$PROJECT_DIR/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="${UV_PROJECT_ENVIRONMENT:-$PROJECT_DIR/.venv}/bin/python"
[ -x "$VENV_PY" ] || die "uv sync 报成功但找不到 venv 里的 python,检查 .venv 位置"
ok "环境已就绪:$VENV_PY"

# ============================ 8. 验证 ============================
say "8/8 验证"

"$VENV_PY" - <<PY || die "校验失败"
import sys, torch
print('  torch     :', torch.__version__)
print('  cuda      :', torch.version.cuda)
print('  python    :', sys.version.split()[0])
print('  算力      :', torch.cuda.get_device_capability())
from torch.utils.cpp_extension import CUDA_HOME
print('  torch 眼中的 CUDA_HOME :', CUDA_HOME)
assert torch.cuda.is_available(), 'CUDA 不可用'

# 真正跑一次 GPU 运算,别只看 is_available()。
# 若 wheel 没编进你这张卡的算力,这里才会暴露
# "no kernel image is available for execution"。
# (cu128 的 wheel 不带 sm_89 内核,4090 上靠 PTX JIT 兜 —— 所以这一步必须真跑。)
a = torch.randn(512, 512, device='cuda'); b = a @ a
torch.cuda.synchronize()
print('  GPU 矩阵乘 :', tuple(b.shape), '✓')
PY

# ── _C 必须真的编出来了 ────────────────────────────────────────────
# 这是本脚本最要紧的一条断言。upstream setup.py 在找不到 nvcc 时会
# 【静默 return None】不编译任何扩展,而 pip/uv 都会报"安装成功"。
# 而运行期 ms_deform_attn.py:334 只看张量在不在 GPU,不看 _C 导没导进来,
# 所以症状会延迟到第一次前向才炸成 NameError —— 那时你已经在跑训练了。
# 必须先 import torch:_C.so 是 setuptools 老后端编出来的,不带 RPATH,
# 靠 torch 先把 libc10/libtorch 载进进程;单独 import _C 会报
# "libc10.so: cannot open shared object file"。
"$VENV_PY" - <<'PY' || die "_C 没编出来 —— 见上面报错。
       若提示 Failed to load custom C++ ops / NameError,说明 nvcc 没被认到。
       回第 6 步看 CUDA_HOME;实在不行用 setup_cloud_conda.sh。"
import torch
from groundingdino import _C
print('  _C        :', _C.__file__)
PY

# 最后确认一遍 .so 里真的含本机算力,而不是捡了上一台机器留下的
_so_now="$(find "$PROJECT_DIR/GroundingDINO" -maxdepth 3 -name '_C*.so' 2>/dev/null | head -1)"
if [ -n "$_so_now" ]; then
  # ⚠️ 不能因为缺 cuobjdump 就【静默跳过】这一步 —— 这是"装成功"和"真编对了"
  #    之间唯一能被自动发现的地方,跳过等于把风险留到训练第一次前向。
  #    cuobjdump 在 conda 里是单独的 cuda-cuobjdump 包,不跟着 cuda-nvcc 走。
  if ! command -v cuobjdump >/dev/null 2>&1; then
    warn "找不到 cuobjdump,【跳过了】_C.so 的算力校验 —— 这条别不当回事。
         它属于单独的 conda 包,装上后手动补验:
           conda install -p ${CUDA_HOME:-<cuda-env>} cuda-cuobjdump
           cuobjdump --list-elf $_so_now | grep sm_${GPU_ARCH}"
  elif cuobjdump --list-elf "$_so_now" 2>/dev/null | grep -q "sm_${GPU_ARCH}"; then
    ok "_C.so 含 sm_${GPU_ARCH}(与当前卡匹配)"
  else
    warn "_C.so 里没看到 sm_${GPU_ARCH} —— 若训练时报 no kernel image,删掉重编:
         rm -f $_so_now && bash $(basename "$0")"
  fi
fi

printf '\n%s  环境就绪 ✓%s\n' "$C_G" "$C_0"
cat <<EOF
  venv   : $VENV_PY
  激活   : source $(dirname "$(dirname "$VENV_PY")")/bin/activate
  锁定   : uv.lock(完整 URL + sha256)
  重建   : uv sync --frozen --no-build-isolation-package groundingdino

  ⚠️ 还没完:本脚本只证明了环境能装、_C 能编、GPU 能算。
     训练本身【至今没有验证过】,请务必实际跑一次:
         $VENV_PY run_grounding.py --num-images 20
EOF
