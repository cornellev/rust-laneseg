#!/usr/bin/env python3
"""Export ADNet to the two ONNX graphs lane-seg loads.

    backbone.onnx  img [1,3,H,W] (BGR/255)
                   -> hm [1,1,h,w], shape_hm [1,1,h,w], batch_features [1,C,h',w']
    roi_head.onnx  batch_features, prior_xs [1,K,S] (model-input pixels)
                   -> cls_logits [1,K,2], reg [1,K,S+1]

Between the two, lane-seg does ADNet's `decode_theta_xy` (a data-dependent
top-k that does not export) in Rust; after the second it does the NMS and
lane decoding. See src/adnet.rs.

The split matters: `SPGHead` still has trained layers *after* the start-point
decode (`pool_prior_features`: grid_sample -> 9x1 conv + BN -> Linear ->
LayerNorm, then `cls_layer` / `reg_layer`). An export that stops at
`batch_features` leaves those weights behind, so both files are needed.

Runs on a laptop CPU: no CUDA, no compiled mmcv ops, no ADNet NMS extension.
mmcv's three inference-time pieces are shimmed when mmcv is not installed, and
the deformable conv is re-expressed with `grid_sample` (ONNX GridSample,
opset 16+), which ONNX Runtime and TensorRT both run natively, so no custom
symbolic or TensorRT plugin is needed.

    python tools/export_adnet_onnx.py \\
        --adnet-repo ~/ADNet \\
        --config ~/ADNet/configs/adnet/culane/resnet18_culane.py \\
        --weights culane_res18.pth \\
        --out-dir models

Requires: torch, torchvision, onnx, onnxruntime (for the check), addict,
yapf, timm, opencv-python, scipy. Omit --weights for random weights.
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


# ── deformable conv as plain ONNX ops ──────────────────────────────────────
def deform_conv2d(x, offset, weight, bias, stride, padding, dilation):
    """DCNv1 (mmcv / torchvision semantics) via grid_sample + matmul.

    Offsets are laid out `[group][tap][dy, dx]` along channels, as in both
    mmcv and torchvision. Out-of-image samples read zero, matching their
    bilinear interpolation. Assumes conv `groups == 1`, which ADNet uses.
    """
    b, c, h, w = (int(s) for s in x.shape)
    out_c, _, kh, kw = (int(s) for s in weight.shape)
    (sh, sw), (ph, pw), (dh, dw) = stride, padding, dilation
    taps = kh * kw
    ho = (h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    groups = int(offset.shape[1]) // (2 * taps)

    ky, kx = torch.meshgrid(
        torch.arange(kh, dtype=x.dtype) * dh,
        torch.arange(kw, dtype=x.dtype) * dw,
        indexing="ij",
    )
    base_y = (torch.arange(ho, dtype=x.dtype) * sh - ph).view(1, ho, 1) + ky.reshape(taps, 1, 1)
    base_x = (torch.arange(wo, dtype=x.dtype) * sw - pw).view(1, 1, wo) + kx.reshape(taps, 1, 1)

    off = offset.view(b, groups, taps, 2, ho, wo)
    py = base_y + off[:, :, :, 0]  # [b, g, taps, ho, wo]
    px = base_x + off[:, :, :, 1]
    grid = torch.stack((px * (2.0 / (w - 1)) - 1.0, py * (2.0 / (h - 1)) - 1.0), dim=-1)
    grid = grid.view(b * groups, taps * ho, wo, 2)

    sampled = F.grid_sample(
        x.reshape(b * groups, c // groups, h, w),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [b*g, c/g, taps*ho, wo]
    cols = sampled.view(b, c * taps, ho * wo)  # channel-major, tap-minor
    out = torch.matmul(weight.view(out_c, c * taps), cols).view(b, out_c, ho, wo)
    if bias is not None:
        out = out + bias.view(1, out_c, 1, 1)
    return out


class ExportDeformConv(nn.Module):
    """Drop-in for mmcv/torchvision DeformConv2d with an exportable forward."""

    def __init__(self, conv):
        super().__init__()
        self.weight = conv.weight
        self.bias = getattr(conv, "bias", None)
        self.stride = _pair(conv.stride)
        self.padding = _pair(conv.padding)
        self.dilation = _pair(conv.dilation)
        assert getattr(conv, "groups", 1) == 1, "grouped deform conv is not supported"

    def forward(self, x, offset):
        return deform_conv2d(x, offset, self.weight, self.bias,
                             self.stride, self.padding, self.dilation)


# ── shims so ADNet imports without mmcv / its CUDA NMS ─────────────────────
def install_shims():
    try:
        import mmcv.cnn  # noqa: F401
        import mmcv.ops  # noqa: F401
    except Exception:
        _shim_mmcv()
    # adnet/ops/nms needs a compiled extension; inference export never calls it.
    ops = types.ModuleType("adnet.ops")
    ops.nms = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NMS runs in Rust"))
    sys.modules["adnet.ops"] = ops
    sys.modules["adnet.ops.nms"] = ops
    if not torch.cuda.is_available():
        # SPGHead.__init__ calls `.cuda()` on a constant.
        torch.Tensor.cuda = lambda self, *a, **k: self


def _norm(cfg, channels, postfix=""):
    kind = cfg["type"]
    if kind in ("BN", "BN2d", "SyncBN"):
        return "bn" + str(postfix), nn.BatchNorm2d(channels)
    if kind == "GN":
        return "gn" + str(postfix), nn.GroupNorm(cfg["num_groups"], channels)
    if kind == "LN":
        return "ln" + str(postfix), nn.LayerNorm(channels)
    raise NotImplementedError(f"norm {kind}")


class _ConvModule(nn.Module):
    """mmcv.cnn.ConvModule, conv -> norm -> act, same state_dict keys."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias="auto", conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type="ReLU"), inplace=True, **_):
        super().__init__()
        if bias == "auto":
            bias = norm_cfg is None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias=bias)
        self.norm_name = None
        if norm_cfg is not None:
            self.norm_name, norm = _norm(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)
        if act_cfg is not None:
            assert act_cfg["type"] == "ReLU", act_cfg
            self.activate = nn.ReLU(inplace=inplace)
        else:
            self.activate = None

    def forward(self, x, *_, **__):
        x = self.conv(x)
        if self.norm_name:
            x = getattr(self, self.norm_name)(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class _DeformConv2d(nn.Module):
    """mmcv.ops.DeformConv2d parameters (weight only, no bias)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, deform_groups=1, bias=False, **_):
        super().__init__()
        kh, kw = _pair(kernel_size)
        self.stride, self.padding, self.dilation = _pair(stride), _pair(padding), _pair(dilation)
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kh, kw))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x, offset):
        return deform_conv2d(x, offset, self.weight, self.bias,
                             self.stride, self.padding, self.dilation)


def _shim_mmcv():
    mmcv = types.ModuleType("mmcv")
    cnn = types.ModuleType("mmcv.cnn")
    ops = types.ModuleType("mmcv.ops")
    cnn.ConvModule = _ConvModule
    cnn.build_norm_layer = _norm
    ops.DeformConv2d = _DeformConv2d
    mmcv.cnn, mmcv.ops = cnn, ops
    mmcv.track_iter_progress = lambda x: x
    sys.modules.update({"mmcv": mmcv, "mmcv.cnn": cnn, "mmcv.ops": ops})


# ── the two exported graphs ────────────────────────────────────────────────
class Backbone(nn.Module):
    """Detector.forward up to (not including) SPGHead's decode_theta_xy."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, img):
        head = self.net.heads
        fea = self.net.backbone(img)
        fea = self.net.neck(fea)
        z = head.SPGU(fea[head.hm_fmap_id])
        hm, shape_hm = z["hm"], z["shape"]
        # torch.hstack exports to Sequence ops TensorRT rejects; same thing here.
        batch_features, _ = head.ALAU(fea[-1], torch.cat([hm, shape_hm], dim=1))
        return hm, shape_hm, batch_features


class RoiHead(nn.Module):
    """SPGHead.forward from pool_prior_features through cls/reg layers."""

    def __init__(self, head, num_anchors):
        super().__init__()
        self.head = head
        self.num_anchors = num_anchors

    def forward(self, batch_features, prior_xs):
        head = self.head
        # pool_prior_features divides prior_xs in place; keep the input intact.
        feats = head.pool_prior_features(batch_features, self.num_anchors, prior_xs.clone())
        feats = feats.reshape(-1, head.anchor_feat_channels)
        cls_logits = head.cls_layer(feats).reshape(1, self.num_anchors, 2)
        reg = head.reg_layer(feats).reshape(1, self.num_anchors, -1)
        return cls_logits, reg


def build_net(args):
    sys.path.insert(0, str(Path(args.adnet_repo).expanduser().resolve()))
    install_shims()
    from adnet.utils.config import Config
    from adnet.models.registry import build_net as adnet_build_net

    cfg = Config.fromfile(str(Path(args.config).expanduser()))
    cfg.backbone.pretrained = False  # we load full weights; skip ImageNet download
    net = adnet_build_net(cfg)

    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        state = ckpt.get("net", ckpt)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        # A checkpoint trained with torchvision's DeformConv2d carries a bias.
        alau = net.heads.ALAU.conv_adaption
        if "heads.ALAU.conv_adaption.bias" in state and getattr(alau, "bias", None) is None:
            alau.bias = nn.Parameter(torch.zeros(alau.weight.shape[0]))
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(f"weights do not match the config:\n"
                             f"  missing: {missing}\n  unexpected: {unexpected}")
        print(f"loaded {args.weights}")
    else:
        print("no --weights: exporting RANDOM weights (speed tests only)")

    # Whatever class ALAU ended up with, export through grid_sample.
    net.heads.ALAU.conv_adaption = ExportDeformConv(net.heads.ALAU.conv_adaption)
    return cfg, net.eval()


def check_deform_conv():
    """Our DCN vs torchvision's reference kernel on random data."""
    from torchvision.ops import deform_conv2d as tv_deform
    torch.manual_seed(0)
    x = torch.randn(1, 8, 10, 25)
    w = torch.randn(8, 8, 3, 3)
    off = torch.randn(1, 2 * 2 * 9, 10, 25) * 3  # 2 deform groups
    ours = deform_conv2d(x, off, w, None, (1, 1), (1, 1), (1, 1))
    ref = tv_deform(x, off, w, padding=(1, 1))
    err = (ours - ref).abs().max().item()
    assert err < 1e-4, f"grid_sample deform conv differs from torchvision by {err}"
    print(f"deform conv matches torchvision (max err {err:.1e})")


def export(module, inputs, path, input_names, output_names, opset):
    torch.onnx.export(module, inputs, str(path), input_names=input_names,
                      output_names=output_names, opset_version=opset,
                      do_constant_folding=True, dynamo=False)
    print(f"wrote {path}")


def verify(path, feeds, expected):
    import onnxruntime as ort
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feeds)
    for name, ref, out in zip(expected, expected.values(), got):
        err = np.abs(ref - out).max()
        print(f"  {path.name}:{name} max |onnx - torch| = {err:.2e}")
        assert err < 1e-3 * max(1.0, np.abs(ref).max()), f"{name} mismatch"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adnet-repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    check_deform_conv()
    cfg, net = build_net(args)
    head = net.heads
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = torch.rand(1, 3, cfg.img_h, cfg.img_w)
    backbone = Backbone(net).eval()
    with torch.no_grad():
        hm, shape_hm, feats = backbone(img)
    print(f"img {tuple(img.shape)} -> hm {tuple(hm.shape)}, shape_hm {tuple(shape_hm.shape)}, "
          f"batch_features {tuple(feats.shape)}; anchors K={head.num_anchor}, S={head.n_offsets}")
    export(backbone, (img,), out / "adnet_backbone.onnx",
           ["img"], ["hm", "shape_hm", "batch_features"], args.opset)

    # Realistic priors from ADNet's own decode, so the check exercises real
    # sampling positions rather than random ones.
    with torch.no_grad():
        hms = torch.clamp(hm.sigmoid(), 1e-4, 1 - 1e-4)
        starts, thetas = head.decode_theta_xy(hms[0], shape_hm[:, 0][0], topk=head.num_anchor)
        anchors = head.generate_anchors_in_one(starts=starts, angles=thetas)
        prior_xs = anchors[:, 6:].unsqueeze(0).float().contiguous()
        roi = RoiHead(head, head.num_anchor).eval()
        cls_logits, reg = roi(feats, prior_xs)
    export(roi, (feats, prior_xs), out / "adnet_roi_head.onnx",
           ["batch_features", "prior_xs"], ["cls_logits", "reg"], args.opset)

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("onnxruntime not installed; skipping the ONNX vs PyTorch check")
        return
    verify(out / "adnet_backbone.onnx", {"img": img.numpy()},
           {"hm": hm.numpy(), "shape_hm": shape_hm.numpy(), "batch_features": feats.numpy()})
    verify(out / "adnet_roi_head.onnx",
           {"batch_features": feats.numpy(), "prior_xs": prior_xs.numpy()},
           {"cls_logits": cls_logits.numpy(), "reg": reg.numpy()})
    print("ONNX matches PyTorch")


if __name__ == "__main__":
    main()
