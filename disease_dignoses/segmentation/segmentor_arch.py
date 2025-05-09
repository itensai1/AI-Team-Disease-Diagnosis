import segmentation_models_pytorch as smp
from torch import nn



class SegmentorArch(nn.Module):
   
    def __init__(self,num_classes):
        super(SegmentorArch,self).__init__()
        
        # Create the encoder
        self.encoder = smp.encoders.get_encoder(
            "resnet34",
            in_channels=3,
            depth=5,
            weights="imagenet"
        )
        
        # Create the decoder
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            attention_type=None
        )
        
        # Create the segmentation head as a Sequential module
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1)
        )

    def forward(self,x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        return masks
        
    

