import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _infer_spatial_hw(num_nodes):
    side = int(round(math.sqrt(int(num_nodes))))
    if side * side != int(num_nodes):
        raise ValueError(
            f"Unable to infer square spatial grid from num_nodes={num_nodes}. "
            "Expected N = n * n."
        )
    return side, side


def _expand_per_stage(value, num_stages):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [int(v) for v in value]
    else:
        values = [int(value)] * int(num_stages)
    if len(values) != int(num_stages):
        raise ValueError(
            f"Expected {num_stages} stage values, got {len(values)}: {values}"
        )
    return values


def _resolve_group_count(num_channels, max_groups=8):
    upper = min(int(max_groups), int(num_channels))
    for groups in range(upper, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _make_norm_2d(num_channels, norm_type="group"):
    norm_key = str(norm_type).lower()
    if norm_key == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_key == "group":
        return nn.GroupNorm(_resolve_group_count(num_channels), num_channels)
    if norm_key == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported branch_norm_type: {norm_type}")


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


def _align_channels(image, target_channels):
    if target_channels is None:
        return image
    target_channels = int(target_channels)
    current_channels = int(image.shape[1])
    if current_channels == target_channels:
        return image
    if current_channels > target_channels:
        return image[:, :target_channels]
    pad = image.new_zeros(
        image.shape[0],
        target_channels - current_channels,
        image.shape[2],
        image.shape[3],
    )
    return torch.cat([image, pad], dim=1)


def _ensure_spatial_image(
    tensor,
    *,
    height=None,
    width=None,
    expected_channels=None,
    name="tensor",
):
    if tensor is None:
        return None

    if tensor.dim() == 4:
        image = tensor
    elif tensor.dim() == 3:
        batch_size = tensor.shape[0]
        if height is None or width is None:
            if tensor.shape[1] == tensor.shape[2]:
                image = tensor.unsqueeze(1)
            else:
                height, width = _infer_spatial_hw(tensor.shape[1])
                image = tensor.view(batch_size, height, width, tensor.shape[2]).permute(0, 3, 1, 2)
        else:
            if tensor.shape[1] == height and tensor.shape[2] == width:
                image = tensor.unsqueeze(1)
            elif tensor.shape[1] == height * width:
                image = tensor.view(batch_size, height, width, tensor.shape[2]).permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)} and does not match "
                    f"target spatial size {(height, width)}"
                )
    else:
        raise ValueError(f"{name} must have 3 or 4 dimensions, got shape {tuple(tensor.shape)}")

    return _align_channels(image.contiguous(), expected_channels)


def _ensure_coord_image(coord, *, height=None, width=None, name="coord"):
    if coord.dim() == 4:
        if coord.shape[1] != 1:
            raise ValueError(f"{name} must have one channel, got shape {tuple(coord.shape)}")
        return coord
    if coord.dim() == 3:
        batch_size = coord.shape[0]
        if height is None or width is None:
            height, width = _infer_spatial_hw(coord.shape[1])
        if coord.shape[1] != height * width or coord.shape[2] != 1:
            raise ValueError(
                f"{name} expects flattened shape (B, N, 1), got {tuple(coord.shape)} "
                f"for spatial size {(height, width)}"
            )
        return coord.view(batch_size, height, width, 1).permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"{name} must have 3 or 4 dimensions, got shape {tuple(coord.shape)}")


def _flatten_spatial_channels(image):
    return image.permute(0, 2, 3, 1).reshape(image.shape[0], -1, image.shape[1])


def _compute_interface_mask(phi_img, interface_bandwidth_factor=2.5):
    if phi_img is None:
        raise ValueError("phi_img is required to compute an interface mask")

    batch_size, channels, height, width = phi_img.shape
    if channels == 0:
        return phi_img.new_zeros(batch_size, 1, height, width)

    band_width = float(interface_bandwidth_factor) / float(max(height, width))
    phi_distance = phi_img.abs().amin(dim=1, keepdim=True)
    return (phi_distance <= band_width).to(phi_img.dtype)


def _prepare_phi_image(phi_field, num_fractures, *, height=None, width=None, reference=None):
    if phi_field is not None:
        return _ensure_spatial_image(
            phi_field,
            height=height,
            width=width,
            expected_channels=num_fractures,
            name="phi_field",
        )

    if reference is None or height is None or width is None:
        raise ValueError("Need a reference tensor and spatial size when phi_field is missing")
    return reference.new_zeros(reference.shape[0], int(num_fractures), height, width)


def _boundary_mask_from_coords(x_img, y_img):
    return 16.0 * x_img * (1.0 - x_img) * y_img * (1.0 - y_img)


def _apply_boundary_mask_to_basis(V_raw, x_coords, y_coords):
    if V_raw.dim() != 3:
        raise ValueError(f"V_raw must have shape (B, N, m), got {tuple(V_raw.shape)}")

    batch_size, num_nodes, basis_dim = V_raw.shape
    height, width = _infer_spatial_hw(num_nodes)
    x_img = _ensure_coord_image(x_coords, height=height, width=width, name="x_coords")
    y_img = _ensure_coord_image(y_coords, height=height, width=width, name="y_coords")
    boundary_mask = _boundary_mask_from_coords(x_img, y_img)
    boundary_mask = boundary_mask.expand(-1, basis_dim, -1, -1)
    V_img = V_raw.transpose(1, 2).reshape(batch_size, basis_dim, height, width)
    V_img = V_img * boundary_mask
    return V_img.reshape(batch_size, basis_dim, num_nodes).transpose(1, 2).contiguous()


def _assemble_ambient_basis(trunk_features, coeffs):
    # Ambient-space DeepONet:
    #   trunk learns a rich dictionary T in R^{B x N x p}
    #   branch/head only produce small coefficient blocks B in R^{B x m x p}
    #   final tentative basis comes from V_raw = T @ B^T.
    # A-orthonormalization is handled later in jumping/model/interface.py so the
    # network itself only predicts a raw basis spanning the coarse space.
    V_raw = torch.matmul(trunk_features, coeffs.transpose(1, 2))
    return torch.nan_to_num(V_raw, nan=0.0, posinf=0.0, neginf=0.0)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4, activation="gelu"):
        super().__init__()
        hidden = max(8, channels // max(int(reduction), 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.activation = _make_activation(activation)

    def forward(self, x):
        scale = self.pool(x)
        scale = self.activation(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        stride=1,
        norm_type="group",
        use_se=False,
        dropout=0.0,
        activation="gelu",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _make_norm_2d(out_channels, norm_type)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_norm_2d(out_channels, norm_type)
        self.se = SqueezeExcitation(out_channels, activation=activation) if use_se else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = _make_activation(activation)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _make_norm_2d(out_channels, norm_type),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        out = self.dropout(out)
        return self.activation(out + residual)


class ResidualNodeBlock(nn.Module):
    def __init__(
        self,
        width,
        *,
        hidden_multiplier=2,
        dropout=0.0,
        use_layernorm=True,
        use_film=True,
        latent_dim=None,
        activation="gelu",
    ):
        super().__init__()
        hidden_dim = int(width * hidden_multiplier)
        self.use_film = bool(use_film and latent_dim is not None)
        self.norm1 = nn.LayerNorm(width) if use_layernorm else nn.Identity()
        self.fc1 = nn.Linear(width, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.film = nn.Linear(latent_dim, 2 * width) if self.use_film else None
        self.gate = nn.Linear(latent_dim, width) if latent_dim is not None else None
        self.activation = _make_activation(activation)

    def forward(self, x, branch_latent=None):
        residual = x
        out = self.norm1(x)
        if self.film is not None and branch_latent is not None:
            scale, shift = self.film(branch_latent).chunk(2, dim=-1)
            out = out * (1.0 + 0.1 * scale.unsqueeze(1)) + 0.1 * shift.unsqueeze(1)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.dropout(out)
        if self.gate is not None and branch_latent is not None:
            gate = torch.sigmoid(self.gate(branch_latent)).unsqueeze(1)
            out = out * gate
        return residual + out


class ResidualLatentBlock(nn.Module):
    def __init__(
        self,
        width,
        *,
        hidden_multiplier=2,
        dropout=0.0,
        use_layernorm=True,
        context_dim=None,
        use_gating=True,
        activation="gelu",
    ):
        super().__init__()
        hidden_dim = int(width * hidden_multiplier)
        self.norm1 = nn.LayerNorm(width) if use_layernorm else nn.Identity()
        self.fc1 = nn.Linear(width, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, width)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = _make_activation(activation)
        self.gate = None
        if use_gating:
            gate_dim = int(context_dim if context_dim is not None else width)
            self.gate = nn.Linear(gate_dim, width)

    def forward(self, x, context=None):
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.dropout(out)
        if self.gate is not None:
            gate_source = x if context is None else context
            out = out * torch.sigmoid(self.gate(gate_source))
        return residual + out


class RichAmbientTrunk(nn.Module):
    """
    Rich trunk for ambient-space DeepONet.

    The trunk no longer emits the final m columns directly. Instead it learns a
    richer ambient dictionary T in R^{B x N x p}, which is later combined with
    small branch-generated coefficients. The resulting basis is A-orthonormalized
    downstream in jumping/model/interface.py instead of inside the network.
    """

    def __init__(
        self,
        *,
        num_fractures,
        p_dim=64,
        trunk_width=256,
        trunk_depth=4,
        trunk_num_fourier_freqs=6,
        trunk_use_film=True,
        trunk_use_layernorm=True,
        trunk_dropout=0.0,
        trunk_use_grad_phi=False,
        trunk_use_interface_band=True,
        interface_bandwidth_factor=2.5,
        branch_latent_dim=256,
        trunk_activation="gelu",
        enforce_zero_boundary=True,
    ):
        super().__init__()
        self.num_fractures = int(num_fractures)
        self.p_dim = int(p_dim)
        self.trunk_width = int(trunk_width)
        self.trunk_depth = int(trunk_depth)
        self.trunk_num_fourier_freqs = int(trunk_num_fourier_freqs)
        self.trunk_use_film = bool(trunk_use_film)
        self.trunk_use_layernorm = bool(trunk_use_layernorm)
        self.trunk_use_grad_phi = bool(trunk_use_grad_phi)
        self.trunk_use_interface_band = bool(trunk_use_interface_band)
        self.interface_bandwidth_factor = float(interface_bandwidth_factor)
        self.enforce_zero_boundary = bool(enforce_zero_boundary)
        self.activation = _make_activation(trunk_activation)

        # Keep the config flags above for checkpoint/config compatibility, but
        # only expose the raw coordinates and level-set field as trunk inputs.
        input_dim = 2 + 4 * self.trunk_num_fourier_freqs + self.num_fractures

        self.input_proj = nn.Linear(input_dim, self.trunk_width)
        self.input_norm = nn.LayerNorm(self.trunk_width) if self.trunk_use_layernorm else nn.Identity()
        self.blocks = nn.ModuleList(
            ResidualNodeBlock(
                self.trunk_width,
                dropout=trunk_dropout,
                use_layernorm=self.trunk_use_layernorm,
                use_film=self.trunk_use_film,
                latent_dim=branch_latent_dim if self.trunk_use_film else None,
                activation=trunk_activation,
            )
            for _ in range(max(self.trunk_depth, 1))
        )
        self.output_norm = nn.LayerNorm(self.trunk_width) if self.trunk_use_layernorm else nn.Identity()
        self.output_proj = nn.Linear(self.trunk_width, self.p_dim)

    def _build_inputs(self, x_coords, y_coords, phi_field):
        phi_img = _ensure_spatial_image(
            phi_field,
            expected_channels=self.num_fractures,
            name="phi_field",
        )
        height, width = phi_img.shape[-2:]
        x_img = _ensure_coord_image(x_coords, height=height, width=width, name="x_coords")
        y_img = _ensure_coord_image(y_coords, height=height, width=width, name="y_coords")

        x_flat = _flatten_spatial_channels(x_img)
        y_flat = _flatten_spatial_channels(y_img)
        phi_flat = _flatten_spatial_channels(phi_img)

        features = [x_flat, y_flat]

        if self.trunk_num_fourier_freqs > 0:
            freqs = (
                (2.0 ** torch.arange(
                    self.trunk_num_fourier_freqs,
                    device=x_flat.device,
                    dtype=x_flat.dtype,
                ))
                * math.pi
            ).view(1, 1, -1)
            x_phase = x_flat * freqs
            y_phase = y_flat * freqs
            features.extend([torch.sin(x_phase), torch.cos(x_phase), torch.sin(y_phase), torch.cos(y_phase)])

        features.append(phi_flat)

        boundary_mask = _flatten_spatial_channels(_boundary_mask_from_coords(x_img, y_img))
        return torch.cat(features, dim=-1), boundary_mask

    def forward(self, x_coords, y_coords, phi_field, branch_latent=None):
        trunk_inputs, boundary_mask = self._build_inputs(x_coords, y_coords, phi_field)
        trunk_features = self.input_proj(trunk_inputs)
        trunk_features = self.activation(self.input_norm(trunk_features))

        for block in self.blocks:
            trunk_features = block(trunk_features, branch_latent=branch_latent)

        trunk_features = self.output_norm(trunk_features)
        trunk_features = self.output_proj(trunk_features)
        if self.enforce_zero_boundary:
            trunk_features = trunk_features * boundary_mask
        return trunk_features


class RichBranchEncoder(nn.Module):
    """
    Shared branch encoder.

    It consumes the log-parameter field log(a) together with the raw level-set
    field phi, then produces a compact global latent vector. The latent is
    reused both by the coefficient heads and by trunk FiLM conditioning.
    """

    def __init__(
        self,
        *,
        num_fractures,
        branch_channels=(32, 64, 128, 128),
        branch_blocks_per_stage=2,
        branch_latent_dim=256,
        branch_norm_type="group",
        branch_use_se=True,
        branch_dropout=0.0,
        interface_bandwidth_factor=2.5,
        branch_activation="gelu",
        canonical_input_size=64,
    ):
        super().__init__()
        channels = [int(ch) for ch in branch_channels]
        if not channels:
            raise ValueError("branch_channels must contain at least one stage")

        self.num_fractures = int(num_fractures)
        self.branch_latent_dim = int(branch_latent_dim)
        self.interface_bandwidth_factor = float(interface_bandwidth_factor)
        self.canonical_input_size = (
            None if canonical_input_size is None else int(canonical_input_size)
        )
        self.stage_blocks = _expand_per_stage(branch_blocks_per_stage, len(channels))

        in_channels = 1 + self.num_fractures
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=5, stride=1, padding=2, bias=False),
            _make_norm_2d(channels[0], branch_norm_type),
            _make_activation(branch_activation),
        )

        stages = []
        current_channels = channels[0]
        for stage_idx, out_channels in enumerate(channels):
            blocks = []
            for block_idx in range(self.stage_blocks[stage_idx]):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                block_in = current_channels if block_idx > 0 else current_channels
                blocks.append(
                    ResidualConvBlock(
                        block_in,
                        out_channels,
                        stride=stride,
                        norm_type=branch_norm_type,
                        use_se=branch_use_se,
                        dropout=branch_dropout,
                        activation=branch_activation,
                    )
                )
                current_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(current_channels, self.branch_latent_dim),
            _make_activation(branch_activation),
            nn.LayerNorm(self.branch_latent_dim),
            nn.Linear(self.branch_latent_dim, self.branch_latent_dim),
        )

    def _build_inputs(self, a_field, phi_field):
        a_img = _ensure_spatial_image(a_field, expected_channels=1, name="a_field")
        height, width = a_img.shape[-2:]
        phi_img = _prepare_phi_image(
            phi_field,
            self.num_fractures,
            height=height,
            width=width,
            reference=a_img,
        )

        log_a = torch.log10(a_img.clamp(min=1e-12))
        inputs = torch.cat([log_a, phi_img], dim=1)
        if self.canonical_input_size is not None and inputs.shape[-1] != self.canonical_input_size:
            inputs = F.interpolate(
                inputs,
                size=(self.canonical_input_size, self.canonical_input_size),
                mode='bilinear',
                align_corners=False,
            )
        return inputs

    def forward(self, a_field, phi_field=None):
        x = self._build_inputs(a_field, phi_field)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)
        return self.mlp(x)


class RichCoefficientHead(nn.Module):
    """
    Maps the shared branch latent to a small coefficient tensor in ambient space.

    When use_column_embedding=True (default), each of the m output columns has its
    own learnable embedding vector that is concatenated to branch_latent before the
    shared MLP.  The m columns are processed together by reshaping to (B*m, input_dim)
    — no Python loop — so each column is structurally forced to come from a different
    input, breaking the collinearity that arises when all columns share the same context.
    """

    def __init__(
        self,
        *,
        latent_dim,
        out_dim,
        p_dim,
        hidden_dim=128,
        head_depth=3,
        dropout=0.0,
        use_layernorm=True,
        use_gating=True,
        use_column_embedding=True,
        column_embedding_dim=32,
        head_activation="gelu",
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.p_dim = int(p_dim)
        self.use_column_embedding = bool(use_column_embedding and column_embedding_dim > 0)

        if self.use_column_embedding:
            # Init at std=1/sqrt(col_dim) so each column embedding has unit-ish L2 norm,
            # comparable to the branch_latent magnitude and visible after input_proj.
            col_dim = int(column_embedding_dim)
            self.column_embedding = nn.Parameter(
                torch.randn(self.out_dim, col_dim) / (col_dim ** 0.5)
            )
            input_dim = int(latent_dim) + int(column_embedding_dim)
        else:
            self.register_parameter('column_embedding', None)
            input_dim = int(latent_dim)

        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualLatentBlock(
                hidden_dim,
                dropout=dropout,
                use_layernorm=use_layernorm,
                context_dim=input_dim,
                use_gating=use_gating,
                activation=head_activation,
            )
            for _ in range(max(int(head_depth) - 1, 0))
        )
        self.output_norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.activation = _make_activation(head_activation)
        # Per-column output projection: each column gets an independent linear map,
        # guaranteeing structural rank = m at initialization regardless of how similar
        # the shared MLP hidden states are across columns.
        self.output_weight = nn.Parameter(torch.empty(self.out_dim, hidden_dim, self.p_dim))
        self.output_bias = nn.Parameter(torch.zeros(self.out_dim, self.p_dim))
        nn.init.kaiming_uniform_(self.output_weight.view(-1, self.p_dim), a=math.sqrt(5))
        bound = 1.0 / math.sqrt(hidden_dim) if hidden_dim > 0 else 0.0
        nn.init.uniform_(self.output_bias, -bound, bound)

    def forward(self, branch_latent):
        """branch_latent: (B, latent_dim) → coeffs: (B, m, p_dim)"""
        B = branch_latent.shape[0]
        m = self.out_dim

        if self.use_column_embedding:
            col_emb = self.column_embedding.unsqueeze(0).expand(B, -1, -1)   # (B, m, col_dim)
            branch_exp = branch_latent.unsqueeze(1).expand(-1, m, -1)        # (B, m, latent_dim)
            context = torch.cat([branch_exp, col_emb], dim=-1)               # (B, m, input_dim)
        else:
            context = branch_latent.unsqueeze(1).expand(-1, m, -1)           # (B, m, latent_dim)

        in_dim = context.shape[-1]
        flat = context.reshape(B * m, in_dim)

        x = self.activation(self.input_proj(flat))
        for block in self.blocks:
            x = block(x, context=flat)
        x = self.output_norm(x)

        # Per-column projection: (B, m, hidden_dim) × (m, hidden_dim, p_dim) → (B, m, p_dim)
        x_3d = x.reshape(B, m, self.hidden_dim)
        return torch.einsum('bmh,mhp->bmp', x_3d, self.output_weight) + self.output_bias


class InterfaceDeepONet(nn.Module):
    """
    Joint ambient-space DeepONet.

    The trunk learns a parameter-conditioned ambient dictionary and the branch
    only emits a small coefficient tensor. The network returns a tentative basis,
    while A-orthonormalization is handled downstream in jumping/model/interface.py.
    """

    def __init__(
        self,
        num_fractures,
        m_dim=10,
        p_dim=64,
        trunk_width=256,
        trunk_depth=4,
        trunk_num_fourier_freqs=6,
        trunk_use_film=True,
        trunk_use_layernorm=True,
        trunk_dropout=0.0,
        trunk_use_grad_phi=False,
        activation="gelu",
        trunk_activation=None,
        branch_activation=None,
        head_activation=None,
        branch_channels=(32, 64, 128, 128),
        branch_blocks_per_stage=2,
        branch_latent_dim=None,
        branch_hidden_dim=None,
        branch_norm_type="group",
        branch_use_se=True,
        head_hidden_dim=256,
        head_depth=3,
        column_embedding_dim=32,
        use_column_embedding=True,
        orth_eps=1e-8,
        enforce_zero_boundary=True,
        branch_canonical_input_size=64,
    ):
        super().__init__()
        self.num_fractures = int(num_fractures)
        self.m_dim = int(m_dim)
        self.p_dim = int(p_dim)
        self.orth_eps = float(orth_eps)  # kept for config compatibility; unused here
        self.enforce_zero_boundary = bool(enforce_zero_boundary)
        trunk_activation = activation if trunk_activation is None else trunk_activation
        branch_activation = activation if branch_activation is None else branch_activation
        head_activation = activation if head_activation is None else head_activation

        branch_latent_dim = int(
            branch_latent_dim
            if branch_latent_dim is not None
            else (branch_hidden_dim if branch_hidden_dim is not None else 256)
        )
        self.branch = RichBranchEncoder(
            num_fractures=self.num_fractures,
            branch_channels=branch_channels,
            branch_blocks_per_stage=branch_blocks_per_stage,
            branch_latent_dim=branch_latent_dim,
            branch_norm_type=branch_norm_type,
            branch_use_se=branch_use_se,
            branch_activation=branch_activation,
            canonical_input_size=branch_canonical_input_size,
        )
        self.trunk = RichAmbientTrunk(
            num_fractures=self.num_fractures,
            p_dim=self.p_dim,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
            trunk_num_fourier_freqs=trunk_num_fourier_freqs,
            trunk_use_film=trunk_use_film,
            trunk_use_layernorm=trunk_use_layernorm,
            trunk_dropout=trunk_dropout,
            trunk_use_grad_phi=trunk_use_grad_phi,
            branch_latent_dim=branch_latent_dim,
            trunk_activation=trunk_activation,
            enforce_zero_boundary=False,
        )
        self.coefficient_head = RichCoefficientHead(
            latent_dim=branch_latent_dim,
            out_dim=self.m_dim,
            p_dim=self.p_dim,
            hidden_dim=head_hidden_dim,
            head_depth=head_depth,
            use_column_embedding=use_column_embedding,
            column_embedding_dim=column_embedding_dim,
            head_activation=head_activation,
        )
        # .. note::
        #     This architecture differs from the pre-fix version: the coefficient head
        #     now runs per-column (with learnable column_embedding), so output_proj weight
        #     shape changed from (hidden_dim, m*p_dim) to (hidden_dim, p_dim).
        #     Pre-fix checkpoints will fail to load with strict=True; this is expected
        #     and intentional, as those checkpoints suffer from a column-collinearity bug.

    def forward(self, a_field, x_coords, y_coords, phi_field):
        branch_latent = self.branch(a_field, phi_field)
        trunk_features = self.trunk(
            x_coords,
            y_coords,
            phi_field,
            branch_latent=branch_latent,
        )
        coeffs = self.coefficient_head(branch_latent)  # (B, m, p_dim)
        V_raw = _assemble_ambient_basis(trunk_features, coeffs)
        if self.enforce_zero_boundary:
            V_raw = _apply_boundary_mask_to_basis(V_raw, x_coords, y_coords)
        return V_raw
