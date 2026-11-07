# Memory Aid

Memory Aid is a screenless, single-button wearable or handheld camera companion for people living with Alzheimer's disease and other dementias. When the user presses the button, the device photographs what is in front of them, checks whether it matches a familiar place from their caregiver's memory book, and speaks a short, warm, reassuring description aloud.

---

## 1. Parts List

Use this exact hardware list:

- **Raspberry Pi Zero W (1st generation)** — ARMv6 single-core 1 GHz, 512 MB RAM, 2.4 GHz Wi-Fi only, with 40-pin GPIO header pins soldered.
- **Raspberry Pi Camera Module** (v1.3, v2, or v3) connected with the **narrow 22-pin Pi Zero camera ribbon cable** (15-pin on the camera end to 22-pin fine pitch on the Pi Zero end).
- **Momentary push button** (2-leg or 4-leg tactile push button) + 2 female-to-female jumper wires (or soldered leads). No resistor required (uses the Pi's internal pull-up).
- **Micro-USB OTG adapter** (micro-USB male to USB-A female) — **required** to plug a standard USB audio dongle into the Pi Zero W's data port.
- **USB audio dongle** (standard plug-and-play USB sound card with 3.5 mm jack).
- **Small powered speaker or headphones** (with a 3.5 mm audio plug connected to the USB audio dongle).
- **MicroSD card** (8 GB or 16 GB Class 10 / A1 recommended) + card reader for your laptop.
- **USB power bank** + micro-USB cable for power.

---

## 2. Wiring

### GPIO Button Wiring (Pin 9 GND & Pin 11 GPIO17)

Connect one leg of the momentary push button to **Physical Pin 9 (GND)** and the other leg to **Physical Pin 11 (GPIO17)**. Polarity does not matter for a momentary switch.

```text
             Raspberry Pi Zero W — 40-Pin GPIO Header
                  (SD card slot is at the TOP)
                  +-----+-----+
             3V3  |  1  |  2  |  5V
           GPIO2  |  3  |  4  |  5V
           GPIO3  |  5  |  6  |  GND
           GPIO4  |  7  |  8  |  GPIO14
             GND  |  9  | 10  |  GPIO15
   +-----[Pin 9]--+-----+-----+
   |      GPIO17  | 11  | 12  |  GPIO18
   |  +--[Pin 11]-+-----+-----+
   |  |   GPIO27  | 13  | 14  |  GND
   |  |   GPIO22  | 15  | 16  |  GPIO23
   |  |      3V3  | 17  | 18  |  GPIO24
   |  |      ...  | ... | ... |  ...
   |  |           +-----+-----+
   |  |
   |  +-----> [ Momentary Push Button ] <-----+
   +------------------------------------------+
```

### Camera Ribbon Orientation & Micro-USB Ports

```text
  +-------------------------------------------------------+
  | [MicroSD]                                             |
  |                                                       |
  |  [ 40-Pin GPIO Header (Pins 1..40 along top edge) ]   |
  |                                                       |
  |   Raspberry Pi Zero W                            [C]  | <-- Camera CSI Connector:
  |                                                  [S]  |     Slide black clip out gently.
  |                                                  [I]  |     Insert narrow 22-pin ribbon with
  |                                                       |     SILVER CONTACTS FACING DOWN
  |  [Mini-HDMI]          [  USB  ]        [  PWR  ]      |     (toward the green PCB / table),
  +--------------------------+----------------+-----------+     blue backing facing UP.
                             |                |
                             v                v
                    Micro-USB DATA    Micro-USB POWER ONLY
                    Plug Micro-USB    Plug USB Power Bank
                    OTG Adapter +     cable here.
                    USB Audio Dongle
                    here!
```

- **Camera Ribbon Orientation**: On the Pi Zero W board edge, gently slide the black plastic retaining clip outward by 1–2 mm. Insert the narrow 22-pin end of the ribbon cable with the **shiny metal contacts facing the green circuit board** (facing down if the board is flat on a table) and the colored/blue stiffener facing up away from the board. Push the black clip firmly back in.
- **Data vs. Power Micro-USB Ports**:
  - **`USB` (inner micro-USB port, closer to the center)**: Data + Power. Plug your **micro-USB OTG adapter** and **USB audio dongle** into this port.
  - **`PWR` (outer micro-USB port, near the corner)**: Power input only. Plug your **USB power bank** here.

---

## 3. Flash the SD Card

1. Download and open **Raspberry Pi Imager** on your laptop.
2. Click **Choose Device** -> select **Raspberry Pi Zero** (first-generation Zero / Zero W).
3. Click **Choose OS** -> **Raspberry Pi OS (other)** -> **Raspberry Pi OS Lite (32-bit)**.
   - **Important**: You **must** choose **32-bit**. The 64-bit OS will not boot on the first-generation Raspberry Pi Zero W's ARMv6 processor.
4. Click **Next**, then click **Edit Settings** (OS customisation):
   - **Set hostname**: `memoryaid`
   - **Set username and password**: e.g., username `pi` (or your preferred username) and a password.
   - **Configure wireless LAN**: Enter your Wi-Fi SSID, password, and country code.
     - **Important**: The Raspberry Pi Zero W supports **2.4 GHz Wi-Fi only** (it cannot see 5 GHz networks). If your venue Wi-Fi is 5 GHz or uses a captive portal, turn on a **phone hotspot configured for 2.4 GHz ("Maximize Compatibility" on iPhone / 2.4 GHz band on Android)** and enter your hotspot's credentials here.
   - **Services tab**: Check **Enable SSH** (Use password authentication).
5. Save settings and write the card. Insert the card into the Pi Zero W and power it on. First boot takes ~2–3 minutes.

---

## 4. Connect to the Pi

Open a terminal on your laptop (or PowerShell on Windows) and connect over SSH:

```bash
ssh pi@memoryaid.local
```
*(Replace `pi` with the username you chose in Raspberry Pi Imager.)*

### Fallback: Raspberry Pi Connect Lite (when hackathon Wi-Fi blocks `.local` / mDNS)

Many conference and university networks block client-to-client SSH (`memoryaid.local`). If you set up on a phone hotspot first, you can install **Raspberry Pi Connect Lite** so you can always get a shell from any browser:

```bash
sudo apt update && sudo apt install -y rpi-connect-lite
loginctl enable-linger
rpi-connect on
rpi-connect signin
```

Open the verification URL printed in the terminal on your laptop, sign in, and you can open a remote shell anytime from `https://connect.raspberrypi.com`.

---

## 5. Copy the Project Over

From your **laptop terminal** (run from the parent folder containing `memoryaid/`):

**macOS / Linux:**
```bash
scp -r memoryaid pi@memoryaid.local:~/
```

**Windows PowerShell:**
```powershell
scp -r .\memoryaid pi@memoryaid.local:~/
```

---

## 6. Run Setup & Add Your API Key

SSH into the Pi and run the idempotent setup script:

```bash
ssh pi@memoryaid.local
cd ~/memoryaid
chmod +x setup.sh
sudo ./setup.sh
```

Now add your Anthropic API key to `/etc/memoryaid.env`:

```bash
sudo nano /etc/memoryaid.env
```

Set `ANTHROPIC_API_KEY=sk-ant-...`, save (`Ctrl+O`, `Enter`, `Ctrl+X`), and restart the service:

```bash
sudo systemctl restart memoryaid
```

After ~25–30 seconds (the time needed to import `picamera2` on the Zero W), the speaker will say:
> *"Hello. Press the button any time and I'll tell you where you are."*

---

## 7. Test Each Part on Its Own

Before relying on the full service, you can verify each piece of hardware individually on the Pi:

1. **Test the Camera**:
   *(Stop the service first if it is holding the camera: `sudo systemctl stop memoryaid`)*
   ```bash
   rpicam-still -o test.jpg
   ls -lh test.jpg
   ```
2. **Test the Audio Output**:
   ```bash
   espeak-ng "hello" --stdout | aplay
   ```
3. **Test the Button (one-line gpiozero check)**:
   ```bash
   python3 -c "from gpiozero import Button; b = Button(17, pull_up=True); print('Press the button...'); b.wait_for_press(); print('Button pressed!')"
   ```
4. **Start & Watch the Service Logs**:
   ```bash
   sudo systemctl start memoryaid
   journalctl -u memoryaid -f
   ```

---

## 8. Adding Places to the Memory Book

The caregiver sets up familiar places inside `~/memoryaid/memories/`. Each subfolder is one place (up to 10 places, up to 2 photos per place):

```text
~/memoryaid/memories/
  joes_cafe/
    note.txt
    1.jpg
    2.jpg
  davids_bedroom/
    note.txt
    1.jpg
```

### How to Take Good Reference Photos
- **Stand where the person usually stands or sits** when they enter the room or look around.
- **Take a wide shot** showing permanent landmarks (windows, countertops, furniture, framed art, signage) rather than small clutter that moves every day.
- **Use good lighting** so details are clear.
- Save 1 or 2 photos as `.jpg`, `.jpeg`, or `.png` (`1.jpg`, `2.jpg`).

### How to Write a Good `note.txt`
Include four calming elements in 2–3 sentences:
1. **The name of the place** (*"Joe's Cafe"* or *"David's bedroom"*).
2. **Who is there and how they are connected** (*"David is your son, and you live together"* or *"Maria is the barista who knows you well"*).
3. **The familiar routine** (*"You come here every morning at 8 for a black coffee"*).
4. **One comforting visual detail** (*"Your favorite armchair is right by the window"*).

### Copying New Memories Over & Restarting
From your laptop:
```bash
scp -r memories/* pi@memoryaid.local:~/memoryaid/memories/
ssh pi@memoryaid.local "sudo systemctl restart memoryaid"
```
On startup, Memory Aid automatically resizes new or changed reference photos to 512 px max edge and caches them in `memories/.cache/`.

---

## 9. Testing on Your Laptop (Mock Mode & Dry Run)

You can run the entire pipeline on your laptop without a Raspberry Pi (`picamera2` and `gpiozero` are never imported in mock mode):

1. **Dry Run (builds the exact API payload and prints size/blocks without calling the network)**:
   ```bash
   python3 -m memoryaid --mock --dry-run --image path/to/photo.jpg
   ```
2. **Interactive Mock Mode (Enter key = button press, `q` + Enter = quit)**:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   python3 -m memoryaid --mock --image path/to/photo.jpg
   ```
   *(You can also pass a folder of images to `--image` and MockCamera will cycle through them on each press.)*
3. **Run the Unit Test Suite**:
   ```bash
   python3 -m unittest discover -s tests -v
   ```

---

## 10. Troubleshooting

- **Camera not detected (`rpicam-still` fails or logs say `I can't see clearly just now`)**:
  - Power off the Pi completely before touching the ribbon cable.
  - Check both ends of the ribbon cable: on the Pi Zero W side, the **silver contacts must face the green PCB**. On the camera module side, the silver contacts must face the camera PCB (away from the black plastic clip on standard v2/v3 modules).
  - Ensure `/boot/firmware/config.txt` (or `/boot/config.txt`) has `camera_auto_detect=1`.
- **No sound / wrong audio device**:
  - Run `aplay -l` to verify your USB audio dongle is listed. Make sure the OTG adapter is plugged into the port labeled **`USB`** (inner port), not `PWR`.
  - Re-run `sudo ./setup.sh` so `/etc/asound.conf` is updated with the right card number, or set `AUDIO_DEVICE=plughw:1,0` explicitly in `/etc/memoryaid.env`.
  - Check volume with `alsamixer`.
- **Button does nothing**:
  - Verify your wires are on **Pin 9 (GND)** and **Pin 11 (GPIO17)** — counting from the square solder pad at Pin 1 (top-left when the SD card slot is at the top).
  - If using a 4-leg tactile switch, wire across diagonal legs so you are sure you are across the switched pair.
  - Remember that button presses are ignored while the device is still speaking or waiting on a request.
- **`memoryaid.local` not found**:
  - Verify your laptop and the Pi are on the **same 2.4 GHz network**.
  - Check your phone hotspot's connected-devices screen to see the Pi's IP address and run `ssh pi@<IP_ADDRESS>`.
- **Slow responses**:
  - Wait until you hear the startup greeting (*"Hello. Press the button..."*) before pressing the button; importing `picamera2` takes 20–30 s on cold start.
  - Keep `MODEL=claude-haiku-5-5` in `/etc/memoryaid.env`.
  - Keep reference photos to 1–2 per place.
- **API errors (bad key, no credit)**:
  - Check `journalctl -u memoryaid -n 50` for `HTTP 401` (invalid `ANTHROPIC_API_KEY`) or `HTTP 400/429` (insufficient credits / rate limit).
  - Double-check `/etc/memoryaid.env` has no stray spaces or quotes around the key.
- **Service keeps restarting**:
  - Run `journalctl -u memoryaid -f` to inspect the startup traceback.
  - Verify `WorkingDirectory` and `User` in `/etc/systemd/system/memoryaid.service` match your username and folder path (`sudo ./setup.sh` sets these automatically).

---

## 11. Switching Audio to an I2S Amp or Bluetooth

Memory Aid pipes `espeak-ng --stdout` into `aplay -q -D <AUDIO_DEVICE>`, so swapping audio hardware only requires changing ALSA configuration or `AUDIO_DEVICE` in `/etc/memoryaid.env`:

### Option A: MAX98357A I2S Mono Amp Breakout
1. Wire the MAX98357A to the Pi Zero W:
   - `VIN` -> Pin 2 (5V), `GND` -> Pin 6 (GND)
   - `BCLK` -> Pin 12 (GPIO18), `LRCLK` -> Pin 35 (GPIO19), `DIN` -> Pin 40 (GPIO21)
2. Add this line to `/boot/firmware/config.txt` (or `/boot/config.txt`):
   ```ini
   dtoverlay=max98357a
   ```
3. Reboot (`sudo reboot`), verify the card number with `aplay -l`, and set `AUDIO_DEVICE=plughw:max98357a,0` (or update `/etc/asound.conf`).

### Option B: Bluetooth Speaker (`bluealsa`)
1. Install `bluez-alsa-utils` (`sudo apt install -y bluez-alsa-utils`), pair and trust your speaker with `bluetoothctl`.
2. Set `AUDIO_DEVICE=bluealsa:DEV=XX:XX:XX:XX:XX:XX,PROFILE=a2dp` in `/etc/memoryaid.env` and restart `memoryaid`.

---

## 12. Demo Tips

- **Print large, full-page color photos** (or display them on a tablet/laptop screen with low glare) of `joes_cafe` and `davids_bedroom` so you can point the camera at them right on the stage table. Also point the camera at the live room to show how it calmly describes an unfamiliar place!
- **Restart the service 2 minutes before going on stage** (`sudo systemctl restart memoryaid`) and wait for the *"Hello..."* greeting so `picamera2` is already warmed up in RAM.
- **Bring a fully charged USB power bank** and run the Pi off your **phone's 2.4 GHz hotspot** so stage Wi-Fi congestion cannot slow down your demo.

---

## 13. Privacy

- **No photos stored on disk by default**: Captured camera frames are resized and base64-encoded purely in RAM (`SAVE_LAST_CAPTURE=false`) and discarded immediately after each request.
- **Encrypted transit**: Photos and caregiver notes are sent over HTTPS (`https://api.anthropic.com/v1/messages`) only when the user physically presses the button.
- **No secret logging**: Neither `ANTHROPIC_API_KEY` nor base64 image payloads are ever written to logs.

---

## 14. Decisions Made

Where details were left open in the specification, the following simple, reliable choices were made:

1. **Place folders without reference photos**: The repository ships with `memories/joes_cafe/note.txt` and `memories/davids_bedroom/note.txt` and no bundled photos (per Section 6). `load_memories()` loads place folders even when they have 0 photos (emitting only their text block and 0 image blocks) so that `--dry-run` works out-of-the-box against the shipped example folders, and automatically includes up to 2 cached reference image blocks as soon as `.jpg`/`.jpeg`/`.png` files are dropped into those folders.
2. **Cache storage format & key**: Reference image cache files are stored as compact JSON files inside `memories/.cache/<sha256_of_relpath>.json` containing `mtime_ns`, `size`, and `b64`. Stale cache files for deleted photos are automatically pruned on startup.
3. **USB audio mixer control setting**: Different USB audio dongles expose their primary volume slider under different ALSA mixer names (`Speaker`, `PCM`, `Master`, or `Headphone`). `setup.sh` iterates through all four common mixer control names and sets whichever exist on the detected USB card to `85% unmute`, wrapping the hardware device in ALSA `type plug` in `/etc/asound.conf` so mono 22.05 kHz WAV output from `espeak-ng` is automatically converted to stereo 44.1/48 kHz if required by the USB DAC.
4. **Zero `pip` packages**: No `pip` packages are used anywhere. Configuration parsing (`.env`), base64 encoding, threading, logging, and subprocess piping are implemented with the Python 3.9+ standard library plus the specified `apt` packages (`python3-requests`, `python3-pil`, `python3-picamera2`, `python3-gpiozero`, `python3-lgpio`).
5. **Env file precedence**: Configuration loads `/etc/memoryaid.env` first, then `./memoryaid.env` (or `--env-file`), and finally overrides with live environment variables (`os.environ`) so systemd deployments and laptop CLI overrides both work seamlessly.
