#!/usr/bin/env bash
set -euo pipefail

echo "==> [1/7] Checking permissions and system architecture..."
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Please run setup.sh with sudo: sudo ./setup.sh" >&2
    exit 1
fi

TARGET_USER="${SUDO_USER:-$(logname 2>/dev/null || echo pi)}"
TARGET_UID="$(id -u "$TARGET_USER" 2>/dev/null || echo 1000)"
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

echo "==> [2/7] Updating apt and installing required system & audio packages..."
apt-get update
# Do NOT install bluez-alsa-utils if PipeWire or PulseAudio is already managing Bluetooth audio,
# because installing bluez-alsa-utils can break an existing working Bluetooth speaker setup.
BASE_PKGS=(
    python3-picamera2
    python3-gpiozero
    python3-lgpio
    python3-requests
    python3-pil
    espeak-ng
    alsa-utils
    bluez
)
apt-get install -y --no-install-recommends "${BASE_PKGS[@]}"

if ! command -v pipewire >/dev/null 2>&1 && ! command -v pulseaudio >/dev/null 2>&1; then
    echo "    Neither PipeWire nor PulseAudio detected; installing bluez-alsa-utils for headless Bluetooth audio..."
    apt-get install -y --no-install-recommends bluez-alsa-utils || true
fi

echo "==> [3/7] Adding user '$TARGET_USER' to hardware access groups (video, gpio, audio, bluetooth)..."
for grp in video gpio audio bluetooth; do
    if getent group "$grp" >/dev/null; then
        usermod -aG "$grp" "$TARGET_USER"
    else
        echo "    WARNING: Group '$grp' does not exist on this system; skipping."
    fi
done

# Enable user lingering so user-session PipeWire/PulseAudio stays active for Bluetooth when running headless
if command -v loginctl >/dev/null 2>&1; then
    loginctl enable-linger "$TARGET_USER" 2>/dev/null || true
fi

echo "==> [4/7] Checking Bluetooth & ALSA audio configuration..."
if command -v rfkill >/dev/null 2>&1; then
    rfkill unblock bluetooth || true
fi
systemctl enable bluetooth.service >/dev/null 2>&1 || true
systemctl start bluetooth.service >/dev/null 2>&1 || true

# Only write /etc/asound.conf if bluealsa is actually running AND PipeWire/PulseAudio are not used
if systemctl is-active --quiet bluealsa 2>/dev/null || systemctl is-active --quiet bluez-alsa 2>/dev/null; then
    if [ ! -f /etc/asound.conf ]; then
        echo "    BlueALSA service is active; creating /etc/asound.conf for Bluetooth audio..."
        cat >/etc/asound.conf <<'EOF'
pcm.!default {
    type asym
    playback.pcm {
        type plug
        slave.pcm "bluealsa"
    }
    capture.pcm {
        type plug
        slave.pcm "bluealsa:PROFILE=sco"
    }
}
EOF
    fi
else
    echo "    Using system default audio routing (PipeWire / PulseAudio / ALSA)."
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
    -e "s|__UID__|${TARGET_UID}|g" \
    -e "s|__WORKDIR__|${PROJECT_DIR}|g" \
    "${PROJECT_DIR}/memoryaid.service" >/etc/systemd/system/memoryaid.service

systemctl daemon-reload
systemctl enable memoryaid.service
systemctl restart memoryaid.service || true

echo "==> [7/7] Setup complete!"
echo ""
echo "NEXT STEPS:"
echo "  1. Edit /etc/memoryaid.env and add your GEMINI_API_KEY:"
echo "       sudo nano /etc/memoryaid.env"
echo "  2. Restart the service:"
echo "       sudo systemctl restart memoryaid"
echo "  3. Watch the live logs:"
echo "       journalctl -u memoryaid -f"
