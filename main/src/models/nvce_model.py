from torch.nn import Module


class NVCE(Module):

    def __init__(self, extractor, n_classes=19, in_channels=3):
        super(NVCE, self).__init__()
        self.n_classes = n_classes
        self.in_channel = in_channels
        self.extractor = extractor

        def forward(self, key_frame, current_frame=None):
            pass
