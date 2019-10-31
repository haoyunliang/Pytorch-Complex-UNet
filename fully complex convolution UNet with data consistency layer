class ComplexUnet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ComplexUnet, self).__init__()

        self.conv1 = ComplexConv2d(in_channel=in_channels, out_channel=64, kernel_size=3, stride=2)                 # 127, 64
        self.conv2 = ComplexConv2d(in_channel=64, out_channel=128, kernel_size=3, stride=2)                         # 63 , 128
        self.conv3 = ComplexConv2d(in_channel=128, out_channel=256, kernel_size=3, stride=2)                        # 31 , 256
        self.conv4 = ComplexConv2d(in_channel=256, out_channel=512, kernel_size=3, stride=2)                        # 15 , 512
        self.conv5 = ComplexConv2d(in_channel=512, out_channel=1024, kernel_size=3, stride=2)                       # 7  , 1024

        self.deconv1 = ComplexConvTranspose2d(in_channel=1024, out_channel=512, kernel_size=3, stride=2)            # 15, 512
        self.deconv2 = ComplexConvTranspose2d(in_channel=512, out_channel=256, kernel_size=3, stride=2)             # 31, 256
        self.deconv3 = ComplexConvTranspose2d(in_channel=256, out_channel=128, kernel_size=3, stride=2)             # 63 , 128
        self.deconv4 = ComplexConvTranspose2d(in_channel=128, out_channel=64, kernel_size=3, stride=2)              # 127, 64
        self.deconv5 = ComplexConvTranspose2d(in_channel=64, out_channel=out_channels, kernel_size=3, stride=2, output_padding=1)     # 255, 12

        self.relu = nn.ReLU()
        self.dc = DataConsistency()

    def forward(self, x, x_k, mask):
        # UNet encoder
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        conv4 = self.relu(self.conv4(conv3))
        conv5 = self.relu(self.conv5(conv4))
        # UNet decoder
        deconv1 = self.relu(self.deconv1(conv5)) + conv4
        deconv2 = self.relu(self.deconv2(deconv1)) + conv3
        deconv3 = self.relu(self.deconv3(deconv2)) + conv2
        deconv4 = self.relu(self.deconv4(deconv3)) + conv1
        deconv5 = self.relu(self.deconv5(deconv4))
        # DC
        output = self.dc(deconv5, x_k, mask)

        return output
