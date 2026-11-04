import torch
import torch.nn as nn
import torch.nn.functional as F


def _canonical_activation_name(activation):
    act = str(activation).lower()
    if act in {"gelu"}:
        return "gelu"
    if act in {"relu"}:
        return "relu"
    if act in {"silu", "swish"}:
        return "silu"
    if act in {"tanh"}:
        return "tanh"
    if act in {"leaky_relu", "lrelu", "leaky"}:
        return "leaky_relu"
    raise ValueError(
        f"Unsupported activation={activation!r}. "
        "Expected one of {'gelu', 'relu', 'silu', 'tanh', 'leaky_relu'}."
    )


def _make_activation(activation):
    act = _canonical_activation_name(activation)
    if act == "gelu":
        return nn.GELU()
    if act == "relu":
        return nn.ReLU()
    if act == "silu":
        return nn.SiLU()
    if act == "tanh":
        return nn.Tanh()
    if act == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    raise AssertionError(f"Unhandled activation {activation!r}")


class BandedCrossModeSpectralConv2d(nn.Module):
    """
    带有低秩带状跨模态混合的 2D 傅里叶谱卷积层 (DEN 论文的 2D 扩展)
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, bandwidth=2, rank=4):
        super(BandedCrossModeSpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)

        # 1. 低秩参数化 W = U * V^T (分为两个角落: 正频率和负频率区域)
        # U 和 V 的形状: (out_c, in_c, modes_out_x, modes_out_y, rank)
        self.U1 = nn.Parameter(self.scale * torch.randn(out_channels, in_channels, modes1, modes2, rank, dtype=torch.cfloat))
        self.V1 = nn.Parameter(self.scale * torch.randn(out_channels, in_channels, modes1, modes2, rank, dtype=torch.cfloat))

        self.U2 = nn.Parameter(self.scale * torch.randn(out_channels, in_channels, modes1, modes2, rank, dtype=torch.cfloat))
        self.V2 = nn.Parameter(self.scale * torch.randn(out_channels, in_channels, modes1, modes2, rank, dtype=torch.cfloat))

        # 2. 构建 2D 带状掩膜 (Band Mask) - vectorized
        i = torch.arange(modes1)
        j = torch.arange(modes2)
        mask = (
            (i.view(modes1, 1, 1, 1) - i.view(1, 1, modes1, 1)).abs() <= bandwidth
        ) & (
            (j.view(1, modes2, 1, 1) - j.view(1, 1, 1, modes2)).abs() <= bandwidth
        )
        self.register_buffer('band_mask', mask.float())

    def forward(self, x):
        batchsize = x.shape[0]
        H, W_dim = x.shape[-2], x.shape[-1]

        # Clip modes to the frequencies actually present in this grid.
        # self.modes1/modes2 are the *maximum* trained modes; at smaller n we
        # just use fewer.  At larger n the stored parameters cover more of the
        # spectrum than the old fixed-12 regime did.
        modes_h = min(self.modes1, H // 2)
        modes_w = min(self.modes2, W_dim // 2 + 1)

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, H, W_dim // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        mask = self.band_mask[:modes_h, :modes_w, :modes_h, :modes_w]

        # --- upper-left (positive-frequency quadrant) ---
        U1 = self.U1[:, :, :modes_h, :modes_w, :]
        V1 = self.V1[:, :, :modes_h, :modes_w, :]
        W1 = torch.einsum('oixyr,oiuvr->oixyuv', U1, torch.conj(V1)) * mask
        out_ft[:, :, :modes_h, :modes_w] = \
            torch.einsum('oixyuv,biuv->boxy', W1, x_ft[:, :, :modes_h, :modes_w])

        # --- lower-left (negative-frequency quadrant) ---
        U2 = self.U2[:, :, :modes_h, :modes_w, :]
        V2 = self.V2[:, :, :modes_h, :modes_w, :]
        W2 = torch.einsum('oixyr,oiuvr->oixyuv', U2, torch.conj(V2)) * mask
        out_ft[:, :, -modes_h:, :modes_w] = \
            torch.einsum('oixyuv,biuv->boxy', W2, x_ft[:, :, -modes_h:, :modes_w])

        return torch.fft.irfft2(out_ft, s=(H, W_dim))


class InterfaceFNO(nn.Module):
    """
    基于 FNO 的算子学习架构
    输入：a_field, x_coords, y_coords, phi_field
    输出：参数依赖的 tentative coarse basis V

    phi_field shape: (Batch, n*n, num_fractures) - flattened from (Batch, num_fractures, n, n)
    """
    def __init__(
        self,
        num_fractures,
        m_dim=10,
        modes=64,
        width=32,
        a_threshold=100.0,
        activation="gelu",
        enforce_zero_boundary=True,
    ):
        super(InterfaceFNO, self).__init__()
        self.m_dim = m_dim
        self.modes = modes
        self.width = width
        self.a_threshold = a_threshold
        self.enforce_zero_boundary = bool(enforce_zero_boundary)
        self.activation = _make_activation(activation)

        # 提升层 P - 输入通道数在 forward 中动态设置
        # self.p = None
        self.in_channels = 3 + num_fractures
        self.p = nn.Conv2d(self.in_channels, self.width, 1)

        # 2. 堆叠的傅里叶卷积块
        self.conv0 = BandedCrossModeSpectralConv2d(width, width, modes, modes)
        self.conv1 = BandedCrossModeSpectralConv2d(width, width, modes, modes)
        self.conv2 = BandedCrossModeSpectralConv2d(width, width, modes, modes)
        self.conv3 = BandedCrossModeSpectralConv2d(width, width, modes, modes)

        # 3. 局部线性旁路 (对应图中的 \mathcal{W})
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        # 4. 投影层 Q (对应图中的 \mathcal{Q})
        self.q = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            _make_activation(activation),
            nn.Conv2d(128, m_dim, 1)
        )

    def forward(self, a_field, x_coords, y_coords, phi_field):
        batch_size = a_field.shape[0]
        n = a_field.shape[-1]

        # --- 数据格式对齐 (转化为图像通道拼接格式) ---
        x_img = x_coords.view(batch_size, 1, n, n)
        y_img = y_coords.view(batch_size, 1, n, n)
        log_a = torch.log10(a_field.clamp(min=1e-12))

        # print(f"Input shapes - log_a: {log_a.shape}, x_img: {x_img.shape}, y_img: {y_img.shape}, phi_field: {phi_field.shape}")
        inputs = torch.cat([log_a, x_img, y_img,  phi_field], dim=1)

        # 动态创建或调整卷积层输入通道
        in_channels = inputs.shape[1]
        if self.p is None or self.p.in_channels != in_channels:
            self.p = nn.Conv2d(in_channels, self.width, 1).to(inputs.device)

        # --- 核心 FNO 前向传播 ---
        x = self.p(inputs) # 提升到高维特征空间

        x1 = self.conv0(x) + self.w0(x)
        x = self.activation(x1)
        x2 = self.conv1(x) + self.w1(x)
        x = self.activation(x2)
        x3 = self.conv2(x) + self.w2(x)
        x = self.activation(x3)
        x4 = self.conv3(x) + self.w3(x)
        x = self.activation(x4)

        # 降维输出: (Batch, m_dim, n, n)
        out = self.q(x)

        if self.enforce_zero_boundary:
            # --- 施加物理边界条件 (Dirichlet BC) ---
            # 假设域为 [0, 1]x[0, 1]，在 2D 图像层面上直接施加 mask 更加安全
            alpha = 10.0  # alpha 越大，边缘越陡峭，内部平坦区域越广
            mask_x = torch.tanh(alpha * x_img) * torch.tanh(alpha * (1.0 - x_img))
            mask_y = torch.tanh(alpha * y_img) * torch.tanh(alpha * (1.0 - y_img))
            bc_mask = mask_x * mask_y
            out = out * bc_mask

        # --- 重塑为离散点云格式 ---
        # 转换为 (Batch, N_nodes, m_dim)，其中 N_nodes = H * W
        V_raw = out.view(batch_size, self.m_dim, -1).transpose(1, 2)

        # 网络只输出 raw basis；A-orthonormalization 统一在
        # jumping/model/interface.py 中完成。
        return torch.nan_to_num(V_raw, nan=0.0, posinf=0.0, neginf=0.0)
