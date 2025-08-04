class CBAMDenseUNet(DenseUNet):
    def __init__(self, in_channels=3, out_channels=1, base=64):
        super().__init__(in_channels, out_channels, base)
        self.cbam1 = CBAM(base + 4 * 12)
        self.cbam2 = CBAM(base*2 + 4 * 12)
        self.cbam3 = CBAM(base*4 + 4 * 12)
        self.cbam_bottleneck = CBAM(base*8 + 4 * 12)

    def forward(self, x):
        e1 = self.cbam1(self.enc1(x))
        e2 = self.cbam2(self.enc2(self.pool(e1)))
        e3 = self.cbam3(self.enc3(self.pool(e2)))
        b = self.cbam_bottleneck(self.bottleneck(self.pool(e3)))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
