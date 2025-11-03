from flask import Flask, jsonify, send_file, render_template_string
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)

# --- Define the Generator network (adjust to your model) ---
class Generator(nn.Module):
    def __init__(self):
        # Used to inherit the torch.nn Module
        super(G, self).__init__()
        # Meta Module - consists of different layers of Modules
        self.main = nn.Sequential(
                nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
                nn.Tanh()
                )

    def forward(self, input):
        output = self.main(input)
        return output


# --- Initialize and load weights ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nz = 100  # latent vector size
generator = Generator(nz=nz).to(device)
if os.path.exists("generator.pth"):
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()


# --- Route to generate an image ---
@app.route("/generate", methods=["GET"])
def generate_image():
    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake = generator(noise).cpu()

        # Convert tensor to image
        img = vutils.make_grid(fake, normalize=True, scale_each=True)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(img)

        # Save and return
        path = "static/generated/fake.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)
        return send_file(path, mimetype='image/png')


# --- Optional home route with HTML ---
@app.route("/")
def index():
    html = """
    <h2>GAN Image Generator</h2>
    <img id="gen" src="/generate" width="256"><br><br>
    <button onclick="refresh()">Generate New</button>
    <script>
      function refresh() {
        document.getElementById('gen').src = '/generate?rand=' + Math.random();
      }
    </script>
    """
    return render_template_string(html)


if __name__ == "__main__":
    app.run(debug=True)
