# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## Program:
Name: R P Loshini
Register Number: 212225230155

```
import sys
!{sys.executable} -m pip install --force-reinstall --no-cache-dir opencv-python==4.10.0.84
```

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load Input Passport Photo and Sunglass Image
# ---------------------------------------------------------
passport_path = r"C:\Users\acer\OneDrive\Pictures\Screenshots\Passport.png"
sunglass_path = r"C:\Users\acer\OneDrive\Pictures\Screenshots\Sunglasses.png"

img = cv2.imread(passport_path, cv2.IMREAD_COLOR)
sunglass_raw = cv2.imread(sunglass_path, cv2.IMREAD_UNCHANGED)

if img is None:
    raise FileNotFoundError(f"Could not load passport image at: {passport_path}")
if sunglass_raw is None:
    raise FileNotFoundError(f"Could not load sunglass image at: {sunglass_path}")

h_img, w_img = img.shape[:2]

# ---------------------------------------------------------
# 2. Define Eye Region (ROI) Shifted Slightly Right
# ---------------------------------------------------------
# Base ROI box that produced x=[166, 502], y=[192, 323]
base_x1, base_x2 = 166, 502
base_y1, base_y2 = 192, 323

shift_x = 8      # nudge slightly further right (in pixels)
shift_y = 0      # negative = up, positive = down (in pixels)
scale_w = 1.0    # keep current enlarged width
scale_h = 1.0    # keep current enlarged height

cx = ((base_x1 + base_x2) / 2.0) + shift_x
cy = ((base_y1 + base_y2) / 2.0) + shift_y
target_w = max(1, int((base_x2 - base_x1) * scale_w))
target_h = max(1, int((base_y2 - base_y1) * scale_h))

x1 = max(0, int(round(cx - target_w / 2.0)))
y1 = max(0, int(round(cy - target_h / 2.0)))
x2 = min(w_img, x1 + target_w)
y2 = min(h_img, y1 + target_h)
target_w = max(1, x2 - x1)
target_h = max(1, y2 - y1)

# ---------------------------------------------------------
# 3. Create Alpha Mask Without Bottom White/Floor Line
# ---------------------------------------------------------
sunglass_bgr_full = sunglass_raw[:, :, :3]
sg_h, sg_w = sunglass_bgr_full.shape[:2]

has_real_alpha = (
    sunglass_raw.ndim == 3
    and sunglass_raw.shape[2] == 4
    and np.min(sunglass_raw[:, :, 3]) < 250
)

if has_real_alpha:
    alpha_full = sunglass_raw[:, :, 3].astype(np.float32) / 255.0
else:
    # Estimate background color from outer border strips
    border_pixels = np.concatenate([
        sunglass_bgr_full[0:5, :, :].reshape(-1, 3),
        sunglass_bgr_full[-5:, :, :].reshape(-1, 3),
        sunglass_bgr_full[:, 0:5, :].reshape(-1, 3),
        sunglass_bgr_full[:, -5:, :].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    bg_color = np.median(border_pixels, axis=0)

    # Color distance from background; > 28 filters faint studio floor/shadow pixels
    diff = np.linalg.norm(sunglass_bgr_full.astype(np.float32) - bg_color, axis=2)
    fg_mask = (diff > 28.0).astype(np.uint8) * 255

    # Zero out bottom strip and erode thin border halo before feathering
    bottom_strip = max(1, int(0.06 * sg_h))
    fg_mask[-bottom_strip:, :] = 0

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    fg_mask = cv2.erode(fg_mask, kernel_open, iterations=2)

    alpha_full = cv2.GaussianBlur(fg_mask.astype(np.float32) / 255.0, (5, 5), 0)

# Resize sunglass BGR and alpha mask to match the eye ROI
sunglass_resized = cv2.resize(sunglass_bgr_full, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32)
alpha_resized = cv2.resize(alpha_full, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

# Suppress faint bleed pixels so no white line shows below the frames
alpha_resized = np.where(alpha_resized < 0.25, 0.0, alpha_resized)
alpha_3ch = np.dstack([alpha_resized, alpha_resized, alpha_resized])

# ---------------------------------------------------------
# 4. Blend Sunglasses Over Eye Region (Alpha Compositing)
# ---------------------------------------------------------
output = img.copy().astype(np.float32)
roi = output[y1:y2, x1:x2]

# Alpha compositing equation: I_out = alpha * I_fg + (1 - alpha) * I_bg
blended_roi = alpha_3ch * sunglass_resized + (1.0 - alpha_3ch) * roi
output[y1:y2, x1:x2] = blended_roi

final_output = np.clip(output, 0, 255).astype(np.uint8)

output_path = "passport_with_sunglasses.png"
cv2.imwrite(output_path, final_output)
print(f"Saved output to {output_path} | Eye ROI: x=[{x1}, {x2}], y=[{y1}, {y2}]")

# ---------------------------------------------------------
# 5. Visualize Original vs. Final Result
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Passport Photo")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
axes[1].set_title("Final Output with Sunglasses")
axes[1].axis("off")

plt.tight_layout()
plt.show()
```

## Output:
<img width="832" height="499" alt="image" src="https://github.com/user-attachments/assets/3665cd33-a437-4402-9fa7-a7e5bbfb1d0a" />


## Result:
Hence, the project that adds sunglasses to photos using image processing has been completed successfully.
