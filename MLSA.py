import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MLSA(nn.Module):
    def __init__(self, dim, bias):
        super(MLSA, self).__init__()
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.patch_size = 8

    def forward(self, x):
        # 通过卷积层生成隐藏表示
        hidden = self.to_hidden(x)

        # 分解为 q, k, v
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        # 将 q, k 变换为 patch 形式
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)

        # 进行傅里叶变换
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        # 计算输出
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        # 最终的输出
        output = v * out
        output = self.project_out(output)  # 恢复到输入通道数

        return output
if __name__ == '__main__':
    block = MLSA(dim=1, bias=1)
    input = torch.rand(2, 1, 512, 512)
    output = block(input)
    print(input.size())
    print(output.size())