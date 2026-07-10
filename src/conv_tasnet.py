import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLayerNorm(nn.Module):
    """Global Layer Norm over both channel and time dims (gLN) — used
    because this is the non-causal (offline) version of Conv-TasNet."""

    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, hidden_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLayerNorm(hidden_channels)

        padding = (kernel_size - 1) * dilation // 2
        self.depthwise_conv = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size,
            padding=padding, dilation=dilation, groups=hidden_channels,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLayerNorm(hidden_channels)

        self.res_out = nn.Conv1d(hidden_channels, in_channels, 1)
        self.skip_out = nn.Conv1d(hidden_channels, in_channels, 1)

    def forward(self, x):
        y = self.prelu1(self.in_conv(x))
        y = self.norm1(y)
        y = self.prelu2(self.depthwise_conv(y))
        y = self.norm2(y)
        residual = self.res_out(y)
        skip = self.skip_out(y)
        return x + residual, skip


class TemporalConvNet(nn.Module):
    def __init__(self, bottleneck_channels, hidden_channels, kernel_size,
                 num_blocks, num_repeats):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_repeats):
            for x in range(num_blocks):
                dilation = 2 ** x
                self.blocks.append(
                    DepthwiseSeparableConv(
                        bottleneck_channels, hidden_channels,
                        kernel_size, dilation,
                    )
                )

    def forward(self, x):
        skip_sum = 0.0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        return skip_sum


class ConvTasNet(nn.Module):
    def __init__(
        self,
        num_sources=4,             # = config["max_sources"]
        enc_filters=512,           # N
        enc_kernel_size=32,        # L (32 samples @16kHz ~ 2ms; use 16 @8kHz)
        bottleneck_channels=128,   # B
        hidden_channels=512,       # H
        kernel_size=3,             # P
        num_blocks=8,              # X
        num_repeats=3,             # R
        mask_activation="sigmoid",
    ):
        super().__init__()
        self.num_sources = num_sources
        self.enc_filters = enc_filters
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        self.encoder = nn.Conv1d(
            1, enc_filters, enc_kernel_size,
            stride=self.enc_stride, bias=False,
        )
        self.layer_norm = GlobalLayerNorm(enc_filters)
        self.bottleneck = nn.Conv1d(enc_filters, bottleneck_channels, 1)

        self.tcn = TemporalConvNet(
            bottleneck_channels, hidden_channels, kernel_size,
            num_blocks, num_repeats,
        )
        self.mask_prelu = nn.PReLU()
        self.mask_conv = nn.Conv1d(
            bottleneck_channels, num_sources * enc_filters, 1,
        )
        self.mask_activation = mask_activation

        self.decoder = nn.ConvTranspose1d(
            enc_filters, 1, enc_kernel_size,
            stride=self.enc_stride, bias=False,
        )

    def forward(self, mixture):
        """
        mixture: (B, T) or (B, 1, T) raw waveform (this project's dataset
                 yields (1, T) per-sample via torchaudio.load, so DataLoader
                 batches arrive as (B, 1, T) — both shapes are accepted).
        returns: (B, num_sources, T) separated waveforms.
        """
        if mixture.dim() == 3:
            mixture = mixture.squeeze(1)  # (B, 1, T) -> (B, T)
        mixture = mixture.unsqueeze(1)  # -> (B, 1, T) for the conv encoder

        enc_out = self.encoder(mixture)  # (B, N, T')
        T_enc = enc_out.shape[-1]

        y = self.layer_norm(enc_out)
        y = self.bottleneck(y)
        y = self.tcn(y)
        y = self.mask_prelu(y)
        masks = self.mask_conv(y)

        masks = masks.view(mixture.shape[0], self.num_sources,
                            self.enc_filters, T_enc)

        if self.mask_activation == "sigmoid":
            masks = torch.sigmoid(masks)
        elif self.mask_activation == "softmax":
            masks = torch.softmax(masks, dim=1)
        elif self.mask_activation == "relu":
            masks = F.relu(masks)
        else:
            raise ValueError(f"unknown mask_activation {self.mask_activation}")

        masked = enc_out.unsqueeze(1) * masks  # (B, C, N, T')
        B, C, N, T_ = masked.shape
        masked = masked.view(B * C, N, T_)
        decoded = self.decoder(masked)
        decoded = decoded.view(B, C, -1)

        target_len = mixture.shape[-1]
        cur_len = decoded.shape[-1]
        if cur_len > target_len:
            decoded = decoded[..., :target_len]
        elif cur_len < target_len:
            decoded = F.pad(decoded, (0, target_len - cur_len))

        return decoded


if __name__ == "__main__":
    model = ConvTasNet(num_sources=4)
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 1, 16000 * 4)  # matches this project's (B,1,T) convention
        out = model(x)
    print("input:", x.shape, "-> output:", out.shape)
    assert out.shape == (2, 4, 16000 * 4)
    print("OK")