import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    """
    Zero-parameter Temporal Shift Module.
    Shifts part of the channels forward/backward along the temporal dimension.
    Input shape: [N*T, C, H, W]
    """
    def __init__(self, net, num_segments=8, shift_div=8):
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.shift_div = shift_div

    @staticmethod
    def shift(x, num_segments, shift_div):
        nt, c, h, w = x.size()
        n_batch = nt // num_segments
        x = x.view(n_batch, num_segments, c, h, w)

        fold = c // shift_div
        out = torch.zeros_like(x)

        # shift left
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # shift right
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        # not shift
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out.view(nt, c, h, w)

    def forward(self, x):
        x = self.shift(x, self.num_segments, self.shift_div)
        return self.net(x)