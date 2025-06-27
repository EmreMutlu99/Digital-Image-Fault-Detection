from flask import Flask, request, render_template, send_file, Response
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import os
import io
import matplotlib.pyplot as plt

# -----------------------------
# U-Net Model Definition
# -----------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        up4 = self.up4(bn)
        merge4 = torch.cat([up4, d4], dim=1)
        dec4 = self.dec4(merge4)
        up3 = self.up3(dec4)
        merge3 = torch.cat([up3, d3], dim=1)
        dec3 = self.dec3(merge3)
        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, d2], dim=1)
        dec2 = self.dec2(merge2)
        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, d1], dim=1)
        dec1 = self.dec1(merge1)

        return torch.sigmoid(self.final(dec1))


# -----------------------------
# Flask App + Model Init
# -----------------------------
app = Flask(__name__, template_folder="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "unet_dagm_class1.pth"))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully.")


# -----------------------------
# Predict Single Image (from upload)
# -----------------------------
def predict_mask(image):
    image = image.convert("L").resize((256, 256))
    img_np = np.array(image, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0].cpu().squeeze().numpy()

    binary_mask = (pred > 0.03).astype(np.uint8) * 255
    return Image.fromarray(binary_mask)


# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return "No image uploaded.", 400

        image = Image.open(file.stream)
        result_mask = predict_mask(image)

        buf = io.BytesIO()
        result_mask.save(buf, format='PNG')
        buf.seek(0)

        return send_file(buf, mimetype='image/png')

    return render_template("index.html")


@app.route("/preview")
def preview_batch():
    """Batch visualization of multiple test images and predicted masks."""
    test_dir = os.path.join("..", "static", "DAGM_KaggleUpload", "Class1", "Test")
    label_dir = os.path.join(test_dir, "Label")
    batch_size = 10

    test_images = [f for f in sorted(os.listdir(test_dir))
                   if f.lower().endswith(".png") and "_label" not in f]
    n = len(test_images)
    batch = test_images[:batch_size]

    plt.figure(figsize=(10, 3 * len(batch)))

    for i, img_file in enumerate(batch):
        img_path = os.path.join(test_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}_label.PNG"
        label_path = os.path.join(label_dir, label_file)

        # Load input image
        img_pil = Image.open(img_path).convert("L").resize((256, 256))
        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            pred_mask = model(img_tensor)[0].cpu().squeeze().numpy()
        pred_binary = (pred_mask > 0.05).astype(float)

        # Load ground truth if available
        if os.path.exists(label_path):
            label_pil = Image.open(label_path).convert("L").resize((256, 256))
            label_np = (np.array(label_pil, dtype=np.uint8) > 127).astype(float)
        else:
            label_np = np.zeros_like(pred_binary)

        # Plot
        plt.subplot(len(batch), 3, i * 3 + 1)
        plt.imshow(img_np, cmap="gray")
        plt.title(f"{img_file}\nOriginal")
        plt.axis("off")

        plt.subplot(len(batch), 3, i * 3 + 2)
        plt.imshow(label_np, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(len(batch), 3, i * 3 + 3)
        plt.imshow(pred_binary, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return Response(buf.getvalue(), mimetype='image/png')


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    print("✅ Starting Flask server...")
    app.run(debug=True)

# -----------------------------
# U-Net Model Definition
# -----------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        up4 = self.up4(bn)
        merge4 = torch.cat([up4, d4], dim=1)
        dec4 = self.dec4(merge4)
        up3 = self.up3(dec4)
        merge3 = torch.cat([up3, d3], dim=1)
        dec3 = self.dec3(merge3)
        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, d2], dim=1)
        dec2 = self.dec2(merge2)
        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, d1], dim=1)
        dec1 = self.dec1(merge1)

        return torch.sigmoid(self.final(dec1))

# -----------------------------
# Flask App + Model Setup
# -----------------------------
app = Flask(__name__, template_folder="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# Adjust path if needed
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "unet_dagm_class1.pth"))

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully.")

def predict_mask(image):
    """Preprocess input image and generate binary defect mask (float thresholding, resizing etc.)."""
    image = image.convert("L").resize((256, 256))  # convert to grayscale and resize
    img_np = np.array(image, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(img_tensor)[0].cpu().squeeze().numpy()

    pred_binary = (pred_mask > 0.03).astype(np.uint8) * 255  # float threshold like in your script
    return Image.fromarray(pred_binary)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return "No image uploaded.", 400

        image = Image.open(file.stream)
        result_mask = predict_mask(image)

        buf = io.BytesIO()
        result_mask.save(buf, format='PNG')
        buf.seek(0)

        return send_file(buf, mimetype='image/png')

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
