#!/usr/bin/env bash
set -euo pipefail

echo "==> [1/7] Checking permissions and system architecture..."
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Please run setup.sh with sudo: sudo ./setup.sh" >&2
    exit 1
fi

TARGET_USER="${SUDO_USER:-$(logname 2>/dev/null || echo pi)}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    echo "    Detected OS: ${PRETTY_NAME:-Unknown}"
else
    echo "    WARNING: /etc/os-release not found; cannot verify Raspberry Pi OS."
fi

ARCH="$(uname -m)"
if [ "$ARCH" != "armv6l" ] && [ "$ARCH" != "armhf" ]; then
    echo "    WARNING: Expected architecture armv6l/armhf (Raspberry Pi Zero W), but found '$ARCH'. Continuing anyway..."
else
    echo "    Architecture verified: $ARCH"
fi

echo "==> [2/7] Updating apt and installing required system packages..."
apt-get update
apt-get install -y --no-install-recommends \
    python3-picamera2 \
    python3-gpiozero \
    python3-lgpio \
    python3-requests \
    python3-pil \
    espeak-ng \
    alsa-utils

echo "==> [3/7] Adding user '$TARGET_USER' to hardware access groups (video, gpio, audio)..."
for grp in video gpio audio; do
    if getent group "$grp" >/dev/null; then
        usermod -aG "$grp" "$TARGET_USER"
    else
        echo "    WARNING: Group '$grp' does not exist on this system; skipping."
    fi
done

echo "==> [4/7] Detecting and configuring USB audio dongle..."
USB_CARD=""
if command -v aplay >/dev/null 2>&1; then
    # Match a card line containing USB-Audio or USB Audio
    USB_CARD="$(aplay -l 2>/dev/null | awk -F'[: ]+' '/^card [0-9]+:.*USB/ {print $2; exit}' || true)"
fi

if [ -n "$USB_CARD" ]; then
    echo "    Found USB audio device on ALSA card $USB_CARD. Writing /etc/asound.conf..."
    cat >/etc/asound.conf <<EOF
pcm.!default {
    type plug
    slave.pcm "hw:${USB_CARD},0"
}
ctl.!default {
    type hw
    card ${USB_CARD}
}
EOF
    echo "    Setting USB audio volume to ~85%..."
    for control in Speaker PCM Master Headphone; do
        if amixer -c "$USB_CARD" scontrols 2>/dev/null | grep -q "'${control}'"; then
            amixer -c "$USB_CARD" sset "$control" 85% unmute >/dev/null 2>&1 || true
        fi
    done
else
    echo "    WARNING: No USB audio card detected via 'aplay -l'."
    echo "    Plug in your USB audio dongle via the micro-USB OTG adapter (into the 'USB' port) and re-run sudo ./setup.sh."
fi

echo "==> [5/7] Setting up /etc/memoryaid.env..."
if [ ! -f /etc/memoryaid.env ]; then
    cp "${PROJECT_DIR}/memoryaid.env.example" /etc/memoryaid.env
    chmod 600 /etc/memoryaid.env
    echo "    Created /etc/memoryaid.env (permissions 600)."
else
    chmod 600 /etc/memoryaid.env
    echo "    /etc/memoryaid.env already exists; keeping existing configuration."
fi

echo "==> [6/7] Installing and enabling memoryaid.service..."
sed -e "s|__USER__|${TARGET_USER}|g" \
    -e "s|__WORKDIR__|${PROJECT_DIR}|g" \
    "${PROJECT_DIR}/memoryaid.service" >/etc/systemd/system/memoryaid.service

systemctl daemon-reload
systemctl enable memoryaid.service
systemctl restart memoryaid.service || true

echo "==> [7/7] Setup complete!"
echo ""
echo "NEXT STEPS:"
echo "  1. Edit /etc/memoryaid.env and add your ANTHROPIC_API_KEY:"
echo "       sudo nano /etc/memoryaid.env"
echo "  2. Restart the service:"
echo "       sudo systemctl restart memoryaid"
echo "  3. Watch the live logs:"
echo "       journalctl -u memoryaid -f"
echo ""
echo "NOTE: Until you add your ANTHROPIC_API_KEY to /etc/memoryaid.env, the device will speak a reminder that it is not set up yet."
