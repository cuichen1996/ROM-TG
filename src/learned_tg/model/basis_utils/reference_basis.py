"""
Reference-basis column constructors and grid-layout utilities for the
interface (jumping-coefficient) problem.
"""

import numpy as np
import torch


# ── Grid layout helpers ───────────────────────────────────────────────────────

def _infer_grid_side_from_num_nodes(num_nodes):
    n = int(round(float(num_nodes) ** 0.5))
    if n * n != int(num_nodes):
        raise ValueError(f"Expected num_nodes to be a perfect square, got {num_nodes}")
    return n


def _zero_dirichlet_boundary_columns(V):
    batch_size, num_nodes, basis_dim = V.shape
    n = _infer_grid_side_from_num_nodes(num_nodes)
    V_img = V.transpose(1, 2).reshape(batch_size, basis_dim, n, n).clone()
    V_img[:, :, 0, :] = 0.0
    V_img[:, :, -1, :] = 0.0
    V_img[:, :, :, 0] = 0.0
    V_img[:, :, :, -1] = 0.0
    return V_img.reshape(batch_size, basis_dim, num_nodes).transpose(1, 2)


def _flatten_coord_field(coord, batch_size, num_nodes, device, dtype):
    if coord is None:
        return None

    coord = coord.to(device=device, dtype=dtype)
    if coord.dim() == 4:
        if coord.shape[1] != 1:
            raise ValueError(f"Expected 4D coordinate field with one channel, got {tuple(coord.shape)}")
        coord = coord.view(coord.shape[0], -1, 1)
    elif coord.dim() == 3:
        if coord.shape[1] == num_nodes and coord.shape[2] == 1:
            pass
        else:
            raise ValueError(
                f"Expected coordinate field shape (B, N, 1), got {tuple(coord.shape)}"
            )
    elif coord.dim() == 2:
        if coord.shape[0] != num_nodes or coord.shape[1] != 1:
            raise ValueError(
                f"Expected coordinate field shape (N, 1), got {tuple(coord.shape)}"
            )
        coord = coord.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported coordinate tensor shape: {tuple(coord.shape)}")

    if coord.shape[0] == 1 and batch_size > 1:
        coord = coord.expand(batch_size, -1, -1)
    elif coord.shape[0] != batch_size:
        raise ValueError(
            f"Coordinate batch size mismatch: expected {batch_size}, got {coord.shape[0]}"
        )
    return coord.contiguous()


def _broadcast_reference_basis(reference_basis, batch_size, num_nodes, basis_dim, device, dtype):
    ref = reference_basis
    if not torch.is_tensor(ref):
        ref = torch.as_tensor(ref, device=device, dtype=dtype)
    else:
        ref = ref.to(device=device, dtype=dtype)

    if ref.dim() == 2:
        if ref.shape[0] != num_nodes:
            raise ValueError(
                f"Reference basis shape mismatch: expected first dim {num_nodes}, got {tuple(ref.shape)}"
            )
        ref = ref.unsqueeze(0)
    elif ref.dim() != 3:
        raise ValueError(
            f"Reference basis must have shape (N, m) or (B, N, m), got {tuple(ref.shape)}"
        )

    if ref.shape[1] != num_nodes:
        raise ValueError(
            f"Reference basis num_nodes mismatch: expected {num_nodes}, got {ref.shape[1]}"
        )
    if ref.shape[2] < basis_dim:
        raise ValueError(
            f"Reference basis has too few columns: need at least {basis_dim}, got {ref.shape[2]}"
        )

    ref = ref[:, :, :basis_dim].contiguous()
    if ref.shape[0] == 1 and batch_size > 1:
        ref = ref.expand(batch_size, -1, -1)
    elif ref.shape[0] != batch_size:
        raise ValueError(
            f"Reference basis batch mismatch: expected {batch_size}, got {ref.shape[0]}"
        )

    return ref


# ── Reference-column constructors ─────────────────────────────────────────────

def _make_constant_reference_columns(batch_size, num_nodes, device, dtype):
    column = torch.ones(batch_size, num_nodes, 1, device=device, dtype=dtype)
    return _zero_dirichlet_boundary_columns(column)


def _make_high_contrast_reference_columns(a_field, threshold, zero_boundary=True):
    if a_field.dim() == 4:
        mask = (a_field[:, 0] > float(threshold)).to(dtype=torch.float64)
    elif a_field.dim() == 3:
        mask = (a_field > float(threshold)).to(dtype=torch.float64)
    else:
        raise ValueError(f"Unsupported a_field shape for reference basis: {tuple(a_field.shape)}")

    columns = mask.reshape(mask.shape[0], -1, 1)
    if zero_boundary:
        columns = _zero_dirichlet_boundary_columns(columns)
    return columns


def _make_block_reference_columns(batch_size, num_nodes, num_blocks_1d, device, dtype, zero_boundary=True):
    n = _infer_grid_side_from_num_nodes(num_nodes)
    num_blocks_1d = max(1, min(int(num_blocks_1d), n))
    columns = []

    for bi in range(num_blocks_1d):
        i0 = int(round(bi * n / num_blocks_1d))
        i1 = int(round((bi + 1) * n / num_blocks_1d))
        for bj in range(num_blocks_1d):
            j0 = int(round(bj * n / num_blocks_1d))
            j1 = int(round((bj + 1) * n / num_blocks_1d))
            col = torch.zeros(batch_size, n, n, device=device, dtype=dtype)
            col[:, i0:i1, j0:j1] = 1.0
            columns.append(col.reshape(batch_size, num_nodes, 1))

    if not columns:
        return torch.zeros(batch_size, num_nodes, 0, device=device, dtype=dtype)

    basis = torch.cat(columns, dim=-1)
    if zero_boundary:
        basis = _zero_dirichlet_boundary_columns(basis)
    return basis


def _make_low_frequency_reference_columns(
    batch_size,
    num_nodes,
    x_coord,
    y_coord,
    num_columns,
    device,
    dtype,
    zero_boundary=True,
):
    if num_columns <= 0:
        return torch.zeros(batch_size, num_nodes, 0, device=device, dtype=dtype)

    x_coord = _flatten_coord_field(x_coord, batch_size, num_nodes, device, dtype)
    y_coord = _flatten_coord_field(y_coord, batch_size, num_nodes, device, dtype)

    modes = [
        (1, 0), (0, 1), (1, 1), (2, 0), (0, 2),
        (2, 1), (1, 2), (2, 2), (3, 0), (0, 3),
        (3, 1), (1, 3), (3, 2), (2, 3),
    ]

    columns = []
    for kx, ky in modes[:num_columns]:
        if kx > 0 and ky > 0:
            col = torch.sin(kx * np.pi * x_coord) * torch.sin(ky * np.pi * y_coord)
        elif kx > 0:
            col = torch.sin(kx * np.pi * x_coord)
        else:
            col = torch.sin(ky * np.pi * y_coord)
        columns.append(col)

    basis = torch.cat(columns, dim=-1)
    if zero_boundary:
        basis = _zero_dirichlet_boundary_columns(basis)
    return basis


def _make_smoothed_random_reference_columns(
    batch_size,
    num_nodes,
    num_columns,
    device,
    dtype,
    e_family=None,
    op_solver=None,
    smooth_steps=5,
    smooth_omega=0.8,
    zero_boundary=True,
    seed=None,
):
    if num_columns <= 0:
        return torch.zeros(batch_size, num_nodes, 0, device=device, dtype=dtype)

    if e_family is not None:
        from learned_tg.model.losses.base import normalize_snapshot_family
        E = normalize_snapshot_family(e_family, expected_num_nodes=num_nodes)
        if E.shape[0] != batch_size:
            raise ValueError(
                f"e_family batch size mismatch: expected {batch_size}, got {E.shape[0]}"
            )

        E = torch.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0).to(device=device, dtype=dtype)
        num_available = int(E.shape[-1])
        if num_available <= 0:
            return torch.zeros(batch_size, num_nodes, 0, device=device, dtype=dtype)

        take = min(int(num_columns), num_available)
        if seed is None:
            indices = torch.randperm(num_available, device=device)[:take]
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            indices = torch.randperm(num_available, generator=generator)[:take].to(device=device)

        basis = E.index_select(dim=-1, index=indices)
        if zero_boundary:
            basis = _zero_dirichlet_boundary_columns(basis)
        return basis

    if op_solver is None:
        raise ValueError(
            "smoothed_random reference columns require either e_family or a valid op_solver."
        )

    n = _infer_grid_side_from_num_nodes(num_nodes)
    rand_kwargs = {"dtype": dtype}
    if seed is None:
        init_img = torch.randn(batch_size, num_columns, n, n, **rand_kwargs)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        init_img = torch.randn(batch_size, num_columns, n, n, generator=generator, **rand_kwargs)
    init_img = init_img.to(device=device)

    if zero_boundary:
        init_img[:, :, 0, :] = 0.0
        init_img[:, :, -1, :] = 0.0
        init_img[:, :, :, 0] = 0.0
        init_img[:, :, :, -1] = 0.0

    f_zero = torch.zeros_like(init_img)
    with torch.no_grad():
        smooth_img = op_solver.asm_smooth(
            init_img, f_zero, M_smooth=int(smooth_steps), omega=float(smooth_omega)
        )

    basis = smooth_img.reshape(batch_size, num_columns, num_nodes).transpose(1, 2)
    if zero_boundary:
        basis = _zero_dirichlet_boundary_columns(basis)
    return basis
