import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms

MODEL_PATH = "model.pth"
TEST_DIR = r"D:\hacthon\Offroad_Segmentation_Training_Dataset\val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (same as training)
from train_segmentation import SegmentationHead, convert_mask, value_map

n_classes = len(value_map)

model = SegmentationHead(384, n_classes, 18, 32).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Load backbone
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((252, 448)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((252, 448), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    ious = []

    for cls in range(n_classes):
        inter = ((pred==cls)&(target==cls)).sum().float()
        union = ((pred==cls)|(target==cls)).sum().float()
        if union==0: continue
        ious.append((inter/union).item())

    return np.mean(ious)

image_dir = os.path.join(TEST_DIR, "Color_Images")
mask_dir  = os.path.join(TEST_DIR, "Segmentation")

ious = []

for name in os.listdir(image_dir)[:20]:  # visualize 20
    img = Image.open(os.path.join(image_dir, name)).convert("RGB")
    mask = Image.open(os.path.join(mask_dir, name))

    mask = convert_mask(mask)

    img_t = transform(img).unsqueeze(0).to(device)
    mask_t = mask_transform(mask)
    mask_t = (mask_t * 255).long().squeeze(0).to(device)

    with torch.no_grad():
        feat = backbone.forward_features(img_t)["x_norm_patchtokens"]
        out = model(feat)
        out = torch.nn.functional.interpolate(out, size=img_t.shape[2:])

    iou = compute_iou(out, mask_t.unsqueeze(0))
    ious.append(iou)

    pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

    # 📊 VISUALIZATION
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img); plt.title("Image")
    plt.subplot(1,3,2); plt.imshow(mask); plt.title("GT")
    plt.subplot(1,3,3); plt.imshow(pred); plt.title(f"Pred IoU:{iou:.2f}")
    plt.show()

print("🔥 Mean IoU:", np.mean(ious))