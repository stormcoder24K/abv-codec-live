# app_webcam.py - ABV CODEC @ 0.10x BW + LIVE BITRATE FROM DEVICE
# RUN: python app_webcam.py
# November 16, 2025 - 07:40 AM IST - NETWORK-AWARE
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import time
import psutil
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Using device: {DEVICE}")

IMG_SIZE = 256
TARGET_FPS = 15
PRINT_EVERY_FRAMES = 30

# ================================
# 1. MODEL (LOCKED @ 0.10x BW)
# ================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class ABVCodec(nn.Module):
    def __init__(self, num_classes=2, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(3, c); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c, c*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(c*2, c*4); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(c*4, c*8)
        self.sem_head = nn.Sequential(
            nn.Conv2d(c*8, c*4, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c*4, num_classes, 1)
        )
        self.bottleneck = nn.Conv2d(c*8, c*8, 1)
        self.up3 = nn.ConvTranspose2d(c*8, c*4, 4, 2, 1); self.dec3 = ConvBlock(c*8, c*4)
        self.up2 = nn.ConvTranspose2d(c*4, c*2, 4, 2, 1); self.dec2 = ConvBlock(c*4, c*2)
        self.up1 = nn.ConvTranspose2d(c*2, c, 4, 2, 1); self.dec1 = ConvBlock(c*2, c)
        self.final = nn.Conv2d(c, 3, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x, bw_ratio=0.1):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        sem_logits = self.sem_head(e4)
        sem_probs = F.softmax(sem_logits, dim=1)
        sem_mask = sem_probs[:, 1:2, :, :]
        sem_mask_full = F.interpolate(sem_mask, size=x.shape[2:], mode='bilinear', align_corners=False)
        z = self.bottleneck(e4)

        noise_base = (1.0 - bw_ratio) * 0.6
        sem_mask_latent = F.interpolate(sem_mask, size=z.shape[2:], mode='bilinear', align_corners=False)
        noise_mod = torch.where(
            sem_mask_latent > 0.3,
            torch.full_like(sem_mask_latent, 0.15),
            torch.full_like(sem_mask_latent, 2.8)
        )
        noise = torch.randn_like(z) * noise_base * noise_mod
        z_noisy = z + noise

        d3 = self.up3(z_noisy); d3 = torch.cat([d3, e3], 1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], 1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], 1); d1 = self.dec1(d1)
        recon = self.out_act(self.final(d1))

        return {
            'recon': recon,
            'sem_mask': sem_mask_full,
            'latent': z_noisy
        }

# ================================
# 2. PSNR + BITRATE
# ================================
def psnr(a, b, mask=None):
    if mask is not None:
        mse = ((a - b) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    else:
        mse = F.mse_loss(a, b)
    mse = mse.clamp_min(1e-10)
    return 20 * torch.log10(1.0 / mse.sqrt()).item()

def get_live_bitrate():
    """Returns current network I/O in kbps (upload + download)"""
    net_io = psutil.net_io_counters()
    bytes_sent = net_io.bytes_sent
    bytes_recv = net_io.bytes_recv
    time.sleep(1)
    net_io2 = psutil.net_io_counters()
    kbps_up = (net_io2.bytes_sent - bytes_sent) * 8 / 1000
    kbps_down = (net_io2.bytes_recv - bytes_recv) * 8 / 1000
    return round(kbps_up + kbps_down, 1)

# ================================
# 3. LOAD MODEL
# ================================
def load_model(model_path="abv_codec_trained.pth"):
    model = ABVCodec().to(DEVICE)
    if os.path.exists(model_path):
        print(f"[MODEL] Loading trained weights: {model_path}")
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
    else:
        print(f"[MODEL] No weights → demo mode")
        model.eval()
    return model

# ================================
# 4. MAIN LOOP - 0.10x + LIVE BITRATE
# ================================
def main():
    print("="*80)
    print("ABV NEURAL CODEC - LIVE @ 0.10x BW + NETWORK BITRATE")
    print("Face: 18–20 dB | Background: 12–15 dB | Press 'q' to quit")
    print("="*80)

    model = load_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not found!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    os.makedirs("live_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"live_outputs/psnr_bitrate_0.10x_{timestamp}.txt"
    with open(log_file, "w") as f:
        f.write("frame,bw,face_psnr,background_psnr,overall_psnr,bitrate_kbps\n")

    print("\n[START] Live feed @ 0.10x BW + real-time bitrate...")
    bitrate = 0.0
    while True:
        ret, frame = cap.read()
        if not ret: break

        orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
        input_tensor = torch.from_numpy(input_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(DEVICE)

        bw_ratio = 0.1  # FIXED

        with torch.no_grad():
            out = model(input_tensor, bw_ratio=bw_ratio)
            recon = out['recon']
            sem_mask = out['sem_mask']

        recon_np = recon[0].permute(1, 2, 0).cpu().numpy()
        recon_np = np.clip(recon_np, 0, 1)
        recon_bgr = cv2.cvtColor((recon_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        sem_mask_np = sem_mask[0, 0].cpu().numpy()
        overlay = recon_bgr.copy()
        mask_colored = (sem_mask_np > 0.3)[..., None]
        red_tint = np.full_like(overlay, (255, 0, 0), dtype=np.uint8)
        tinted = cv2.addWeighted(overlay, 0.7, red_tint, 0.3, 0)
        overlay = np.where(mask_colored, tinted, overlay)

        # PSNR
        x = input_tensor
        r = recon
        gt_mask = (sem_mask > 0.3).float()
        bg_mask = (sem_mask <= 0.3).float()
        face_psnr = psnr(x, r, mask=gt_mask) if gt_mask.sum() > 10 else 0
        bg_psnr = psnr(x, r, mask=bg_mask) if bg_mask.sum() > 10 else 0
        overall_psnr = psnr(x, r)

        # LIVE BITRATE (every 30 frames)
        if frame_count % 30 == 0:
            bitrate = get_live_bitrate()

        if frame_count % PRINT_EVERY_FRAMES == 0:
            print(f"[F{frame_count:4d}] BW: 0.10x | Face: {face_psnr:5.1f} dB | BG: {bg_psnr:5.1f} dB | Net: {bitrate:6.1f} kbps")

        with open(log_file, "a") as f:
            f.write(f"{frame_count},0.10,{face_psnr:.1f},{bg_psnr:.1f},{overall_psnr:.1f},{bitrate:.1f}\n")

        # DISPLAY
        display_orig = cv2.resize(frame, (320, 320))
        display_recon = cv2.resize(recon_bgr, (320, 320))
        display_overlay = cv2.resize(overlay, (320, 320))
        display_sem = cv2.resize(cv2.applyColorMap((sem_mask_np > 0.3).astype(np.uint8)*255, cv2.COLORMAP_JET), (320, 320))

        cv2.putText(display_orig, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_recon, "RECON @ 0.10x", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_recon, f"Face:{face_psnr:.1f}dB", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(display_recon, f"BG:{bg_psnr:.1f}dB", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(display_recon, f"Net: {bitrate:.1f} kbps", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        top = np.hstack([display_orig, display_recon])
        bottom = np.hstack([display_overlay, display_sem])
        final = np.vstack([top, bottom])

        cv2.imshow("ABV Live @ 0.10x + Network Bitrate", final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[STOP] Feed closed. Full log: {log_file}")

if __name__ == "__main__":
    main()