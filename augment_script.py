import cv2
import albumentations as A
import os
import copy
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_IMG_DIR = "raw_data/train/images"
INPUT_LBL_DIR = "raw_data/train/labels"
OUTPUT_IMG_DIR = "augmented_data/images"
OUTPUT_LBL_DIR = "augmented_data/labels"
COUNT_PER_IMAGE = 20  # 99 * 20 = 1,980 images

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

# Define the Augmentation Pipeline
# We use 'xy' for keypoints to match YOLO's normalized coordinates
transform = A.Compose([
    A.SafeRotate(limit=180, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

image_files = [f for f in os.listdir(INPUT_IMG_DIR) if f.endswith('.jpg')]

print(f"🚀 Starting augmentation for {len(image_files)} images...")
for img_name in tqdm(image_files):
    img_path = os.path.join(INPUT_IMG_DIR, img_name)
    lbl_path = os.path.join(INPUT_LBL_DIR, img_name.replace('.jpg', '.txt'))
    
    if not os.path.exists(lbl_path):
        continue

    image = cv2.imread(img_path)
    if image is None: continue
    h, w = image.shape[:2]

    # Read YOLO Label
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
        if not lines: continue
        
        # We only process the first line (first gauge detected)
        line = lines[0].split()
        
        # SAFETY CHECK: A full pose label with 3 keypoints should have 13 elements:
        # [class, x, y, w, h, kp1_x, kp1_y, kp1_v, kp2_x, kp2_y, kp2_v, kp3_x, kp3_y, kp3_v]
        # In your case, it seems to have at least 13 or 14. We check for at least 11.
        if len(line) < 11:
            print(f"\nSkipping {img_name}: Label only has {len(line)} values (missing keypoints).")
            continue
        
        try:
            kps = [
                (float(line[5]) * w, float(line[6]) * h),
                (float(line[8]) * w, float(line[9]) * h),
                (float(line[11]) * w, float(line[12]) * h)
            ]
        except (ValueError, IndexError):
            print(f"\nError parsing {img_name}. Skipping.")
            continue

    for i in range(COUNT_PER_IMAGE):
        augmented = transform(image=image, keypoints=kps)
        aug_img = augmented['image']
        aug_kps = augmented['keypoints']

        # Save Image
        new_name = f"aug_{i}_{img_name}"
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, new_name), aug_img)

        # Save Label (Normalize back to 0-1)
        with open(os.path.join(OUTPUT_LBL_DIR, new_name.replace('.jpg', '.txt')), 'w') as f:
            # We use the original box coordinates (line[:5]) 
            # Note: For Pose estimation, the box is less critical than the KPs
            new_line = line[:5] 
            for kp in aug_kps:
                new_line.extend([f"{kp[0]/w:.6f}", f"{kp[1]/h:.6f}", "2"])
            f.write(" ".join(new_line) + "\n")
