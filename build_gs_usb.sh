#!/usr/bin/env bash
# ============================================================================
# build_gs_usb.sh — 在 Jetson Thor T5000 (L4T R38.2.1, kernel 6.8.12-tegra)
#                    上编译并安装 gs_usb 内核模块
#
# 背景：JetPack 6.x 的默认内核未编译 gs_usb 模块 (CONFIG_CAN_GS_USB=m 未启用)
#       导致 USB 转 CAN 适配器（如 OpenMoko/Canable/Candlelight 1d50:606f）
#       插入后 Driver=[none]，无法创建 canX 接口
#
# 用法：
#   chmod +x build_gs_usb.sh
#   sudo ./build_gs_usb.sh
#
# 执行完成后，gs_usb.ko 将被安装到系统模块目录，并可通过 modprobe gs_usb 加载
# ============================================================================
set -Eeuo pipefail

die() { echo "❌ ERROR: $*" >&2; exit 1; }
info() { echo "ℹ️  $*"; }
ok()   { echo "✅ $*"; }

# ── 1. 权限检查 ──
[[ $EUID -eq 0 ]] || die "请以 root 执行：sudo $0"

# ── 2. 内核版本 & 头文件目录 ──
KVER=$(uname -r)
# L4T R38 的 Canonical 内核头文件路径
KHEADER_BASE="/usr/src/linux-headers-${KVER}-ubuntu24.04_aarch64"
KBUILD_DIR="${KHEADER_BASE}/3rdparty/canonical/linux-noble"

[[ -f "${KBUILD_DIR}/.config" ]] || die "未找到内核构建目录 ${KBUILD_DIR}/.config\n请确认已安装 nvidia-l4t-kernel-headers"
[[ -f "${KBUILD_DIR}/Module.symvers" ]] || die "未找到 Module.symvers"

info "内核版本: $KVER"
info "构建目录: $KBUILD_DIR"

# ── 3. 安装编译依赖 ──
info "安装编译工具 (如已安装会跳过)..."
apt-get update -qq
apt-get install -y -qq build-essential bc flex bison libssl-dev wget >/dev/null 2>&1
ok "编译依赖已就绪"

# ── 4. 下载 gs_usb.c 源码 ──
# 使用与内核版本 6.8.x 匹配的上游 Linux 源码
# kernel.org v6.8 tag
GS_USB_URL="https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/plain/drivers/net/can/usb/gs_usb.c?h=v6.8"
WORK_DIR="/tmp/gs_usb_build"
mkdir -p "$WORK_DIR"

if [[ -f "${WORK_DIR}/gs_usb.c" ]]; then
    info "gs_usb.c 已存在，跳过下载"
else
    info "从 kernel.org 下载 gs_usb.c (v6.8)..."
    wget -q -O "${WORK_DIR}/gs_usb.c" "$GS_USB_URL" || die "下载 gs_usb.c 失败，请检查网络"
    ok "gs_usb.c 下载完成"
fi

# ── 5. 创建外部模块 Makefile ──
cat > "${WORK_DIR}/Makefile" << 'MAKEFILE_EOF'
obj-m += gs_usb.o

KBUILD_DIR ?= /usr/src/linux-headers-$(shell uname -r)-ubuntu24.04_aarch64/3rdparty/canonical/linux-noble

all:
	$(MAKE) -C $(KBUILD_DIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KBUILD_DIR) M=$(PWD) clean
MAKEFILE_EOF

# ── 6. 编译模块 ──
info "编译 gs_usb.ko..."
cd "$WORK_DIR"
make KBUILD_DIR="$KBUILD_DIR" clean 2>/dev/null || true
make KBUILD_DIR="$KBUILD_DIR" 2>&1 | tail -5

[[ -f "${WORK_DIR}/gs_usb.ko" ]] || die "编译失败：gs_usb.ko 未生成"
ok "编译成功：${WORK_DIR}/gs_usb.ko"

# ── 7. 安装模块 ──
INSTALL_DIR="/lib/modules/${KVER}/kernel/drivers/net/can/usb"
mkdir -p "$INSTALL_DIR"
cp "${WORK_DIR}/gs_usb.ko" "$INSTALL_DIR/"
depmod -a
ok "模块已安装到 $INSTALL_DIR"

# ── 8. 加载模块 ──
info "加载 gs_usb 模块..."
modprobe gs_usb || insmod "${INSTALL_DIR}/gs_usb.ko"
ok "gs_usb 模块已加载"

# ── 9. 验证 ──
if lsmod | grep -q "gs_usb" || lsmod | grep -q "gs-usb"; then
    ok "gs_usb 模块正在运行"
else
    echo "⚠️  lsmod 未检测到 gs_usb，检查 dmesg..."
    dmesg | grep -i gs_usb | tail -5
    die "gs_usb 模块加载失败"
fi

# 检查是否有 USB CAN 设备被识别
sleep 1
CAN_IFACES=$(ip -br link show type can 2>/dev/null | awk '{print $1}' || true)
if [[ -n "$CAN_IFACES" ]]; then
    info "检测到以下 CAN 接口："
    for ifc in $CAN_IFACES; do
        drv=$(ethtool -i "$ifc" 2>/dev/null | awk -F': ' '/^driver:/{print $2}' || echo "unknown")
        bus=$(ethtool -i "$ifc" 2>/dev/null | awk -F': ' '/^bus-info:/{print $2}' || echo "unknown")
        echo "   $ifc  driver=$drv  bus=$bus"
    done
fi

# ── 10. 设置开机自动加载 ──
if ! grep -q "^gs_usb" /etc/modules-load.d/*.conf 2>/dev/null; then
    echo "gs_usb" >> /etc/modules-load.d/gs_usb.conf
    ok "已配置开机自动加载 gs_usb"
fi

echo ""
echo "=============================================="
echo "  gs_usb 驱动安装完成！"
echo "  现在可以插入 USB CAN 适配器并运行："
echo "    ip link show type can"
echo "    sudo ethtool -i canX | grep bus"
echo "=============================================="
