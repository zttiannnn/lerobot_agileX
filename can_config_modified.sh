#!/usr/bin/env bash
set -Eeuo pipefail

# 用法：
#  单个设备：
#    sudo ./can_config_modified.sh --right 1-6:1.0 --rate-right 1000000
#  两个设备：
#    sudo ./can_config_modified.sh --right 1-6:1.0 --left 1-5:1.0 --rate-right 1000000 --rate-left 1000000
#
# 说明：
#  --right         必填，can_right 的 USB 硬件地址（如 1-6:1.0）
#  --left          可选，can_left  的 USB 硬件地址
#  --rate-right    可选，右侧比特率，默认 1000000
#  --rate-left     可选，左侧比特率，默认 1000000
#  --dry-run       只打印将执行的操作，不改系统

RIGHT_BUS=""
LEFT_BUS=""
RATE_RIGHT=1000000
RATE_LEFT=1000000
DRY_RUN=0

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
run()  { if [[ $DRY_RUN -eq 1 ]]; then echo "[DRY] $*"; else eval "$@"; fi; }

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --right)       RIGHT_BUS="${2:-}"; shift 2 ;;
    --left)        LEFT_BUS="${2:-}"; shift 2 ;;
    --rate-right)  RATE_RIGHT="${2:-}"; shift 2 ;;
    --rate-left)   RATE_LEFT="${2:-}"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '1,40p' "$0"; exit 0 ;;
    *)
      die "Unknown arg: $1" ;;
  esac
done

[[ -n "$RIGHT_BUS" ]] || die "必须提供 --right <USB_BUS-ID，比如 1-6:1.0>"

if [[ $EUID -ne 0 && $DRY_RUN -eq 0 ]]; then
  die "请以 root 运行（sudo），或使用 --dry-run 先查看动作"
fi

# 依赖检查
command -v ip >/dev/null || die "缺少 ip (iproute2)"
command -v ethtool >/dev/null || die "缺少 ethtool"
modprobe gs_usb 2>/dev/null || true

# 枚举所有 CAN 网卡
mapfile -t IFACES < <(ip -br link show type can | awk '{print $1}')
[[ ${#IFACES[@]} -gt 0 ]] || die "未检测到任何 CAN 接口"

declare -A BUS2IF=()
for ifc in "${IFACES[@]}"; do
  # 获取驱动和总线信息
  drv=$(ethtool -i "$ifc" 2>/dev/null | awk -F': ' '/^driver:/{print $2}')
  bus=$(ethtool -i "$ifc" 2>/dev/null | awk -F': ' '/^bus-info:/{print $2}')
  [[ -n "$bus" ]] || continue
  BUS2IF["$bus"]="$ifc"
done

[[ ${#BUS2IF[@]} -gt 0 ]] || die "未找到有效的 CAN 设备（bus-info 为空）"

find_iface_by_bus() {
  local bus="$1"
  [[ -n "${BUS2IF[$bus]:-}" ]] || return 1
  echo "${BUS2IF[$bus]}"
}

ensure_name_free() {
  local name="$1" expect="$2"
  if ip link show "$name" >/dev/null 2>&1; then
    # 已存在，若不是我们要改的同一个接口则报错
    [[ "$name" == "$expect" ]] || die "接口名 $name 已被其他网卡占用，请先腾空或改名"
  fi
}

config_one() {
  local target_name="$1" bus="$2" bitrate="$3"

  [[ -n "$bus" ]] || die "缺少 $target_name 的 USB bus"
  local ifc
  if ! ifc=$(find_iface_by_bus "$bus"); then
    die "未找到 bus $bus 对应的 CAN 接口；请确认 ethtool -i canX 的 bus-info"
  fi

  info "$target_name: bus=$bus 接口=$ifc 目标比特率=$bitrate"

  ensure_name_free "$target_name" "$ifc"

  # 如果名字不同，先 down 再改名
  if [[ "$ifc" != "$target_name" ]]; then
    run "ip link set '$ifc' down"
    run "ip link set '$ifc' name '$target_name'"
    ifc="$target_name"
  else
    run "ip link set '$ifc' down"
  fi

  # 设定比特率并 up
  run "ip link set '$ifc' type can bitrate $bitrate"
  run "ip link set '$ifc' up"

  # 校验
  if [[ $DRY_RUN -eq 0 ]]; then
    cur_rate=$(ip -details link show "$ifc" | awk '/bitrate/{print $2}')
    [[ "$cur_rate" == "$bitrate" ]] || die "$ifc 比特率设置失败：当前 $cur_rate, 期望 $bitrate"
  fi

  info "$target_name 已就绪：接口=$ifc 比特率=$bitrate"
}

# 单/双设备配置
config_one "can_right" "$RIGHT_BUS" "$RATE_RIGHT"

if [[ -n "$LEFT_BUS" ]]; then
  [[ "$LEFT_BUS" != "$RIGHT_BUS" ]] || die "--left 不能与 --right 使用相同 bus"
  config_one "can_left" "$LEFT_BUS" "$RATE_LEFT"
else
  info "未提供 --left，跳过 can_left"
fi

info "完成。当前 CAN 接口："
ip -br -details link show type can | sed 's/^/[CAN] /'
