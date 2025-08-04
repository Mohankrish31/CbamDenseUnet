class CBAM_RDB_Block(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.rdb = RDB(in_channels, growth_rate)
        self.cbam = CBAM(in_channels)
    def forward(self, x):
        out = self.rdb(x)
        out = self.cbam(out)
        return out
