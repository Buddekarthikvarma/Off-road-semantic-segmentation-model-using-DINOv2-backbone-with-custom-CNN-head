import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==============================
# 🔧 PATHS
# ==============================

TRAIN_DIR = r"D:\hacthon\Offroad_Segmentation_Training_Dataset\train"
VAL_DIR   = r"D:\hacthon\Offroad_Segmentation_Training_Dataset\val"
SAVE_PATH = r"D:\hacthon\model.pth"

# ==============================
# CONFIG
# ==============================

EPOCHS = 25
BATCH_SIZE = 4
LR = 1e-4

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

n_classes = len(value_map)

# ==============================
# DATASET
# ==============================

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir  = os.path.join(data_dir, "Segmentation")

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Missing: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Missing: {self.mask_dir}")

        self.ids = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.mask_transform = mask_transform

        print(f"Loaded {len(self.ids)} samples from {data_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))

        mask = convert_mask(mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            mask = (mask * 255).long()

        return img, mask.squeeze(0)

# ==============================
# MODEL
# ==============================

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, H, W):
        super().__init__()
        self.H, self.W = H, W

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 7, padding=3, groups=128),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.net(x)

# ==============================
# LOSS FUNCTIONS
# ==============================

def dice_loss(pred, target, smooth=1):
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=n_classes).permute(0,3,1,2)

    intersection = (pred * target_onehot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# ==============================
# MAIN
# ==============================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using:", device)

    # IMAGE SIZE
    w, h = 448, 252

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    # 🔥 IMPORTANT FIX
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    trainset = MaskDataset(TRAIN_DIR, transform, mask_transform)
    valset   = MaskDataset(VAL_DIR, transform, mask_transform)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader   = DataLoader(valset, batch_size=BATCH_SIZE)

    # BACKBONE
    print("Loading DINOv2...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.to(device).eval()

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # GET FEATURE SIZE
    sample = trainset[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        feat = backbone.forward_features(sample)["x_norm_patchtokens"]

    embed_dim = feat.shape[-1]

    model = SegmentationHead(embed_dim, n_classes, h//14, w//14).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    # TRAIN LOOP
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, masks in tqdm(trainloader):
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.no_grad():
                feat = backbone.forward_features(imgs)["x_norm_patchtokens"]

            out = model(feat)
            out = F.interpolate(out, size=imgs.shape[2:])

            # 🔥 COMBINED LOSS
            loss = ce_loss(out, masks) + 0.5 * dice_loss(out, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # VALIDATION
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, masks in valloader:
                imgs, masks = imgs.to(device), masks.to(device)

                feat = backbone.forward_features(imgs)["x_norm_patchtokens"]
                out = model(feat)
                out = F.interpolate(out, size=imgs.shape[2:])

                loss = ce_loss(out, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valloader)
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")

    # SAVE MODEL
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Model saved at: {SAVE_PATH}")

    # PLOT LOSS
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss.png")
    plt.show()


if __name__ == "__main__":
    main()