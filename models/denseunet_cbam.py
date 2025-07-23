# models/cbam_denseunet.py
from models.denseunet import DenseBlock
from models.cbam import cbam  # Assuming you already have CBAM implemented

class cbam_denseunet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = nn.Sequential(DenseBlock(in_channels, 4), cbam(3 + 4 * 32))
        self.enc2 = nn.Sequential(DenseBlock(3 + 4 * 32, 4), cbam(3 + 8 * 32))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(DenseBlock(3 + 8 * 32, 4), cbam(3 + 12 * 32))

        self.up1 = nn.ConvTranspose2d(3 + 12 * 32, 128, 2, stride=2)
        self.dec1 = nn.Conv2d(128 + 3 + 8 * 32, 64, 3, padding=1)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d = self.up1(b)
        d = torch.cat([d, e2], dim=1)
        d = F.relu(self.dec1(d))

        return self.final(d)
