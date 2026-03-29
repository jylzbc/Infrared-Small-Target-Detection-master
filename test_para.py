import torch
from thop import profile, clever_format
from model.model_DNANet import DNANet

model = DNANet(
    num_classes=1,
    input_channels=1,
    nb_filter=[16, 32, 64, 128, 256],
    deep_supervision=False
)

total = sum(p.numel() for p in model.parameters())
print(f"参数量: {total: ,} ({total/1e6:.3f}M)")

input = torch.randn(1, 1, 256, 256)
flops, params = profile(model, inputs=(input,), verbose=False)
flops_f, params_f = clever_format([flops, params], "%.3f")
print(f"FLOPs: {flops_f}")