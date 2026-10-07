"""
The PyTorch port computes what the FER notebook's TensorFlow blocks compute: the TF weights are loaded into the
port and the outputs compared (reference made by tests/fixtures/make_fer_tf_blocks.py). Plus a whole-model shape
test with random weights.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epills.core.models.hybrid import (CBAM, CapsuleFusionBlock, HybridPillNet,  # noqa: E402
                                       TransformerRefinementBlock)

REF = np.load(Path(__file__).parent / "fixtures" / "fer_tf_blocks.npz")
TOL = 1e-5


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def load_dense(layer, kernel, bias):
    layer.weight.data, layer.bias.data = t(kernel.T), t(bias)


@torch.no_grad()
def test_cbam_matches_tf():
    m = CBAM(512).eval()
    w = [REF[f"cbam_w{i}"] for i in range(8)]
    load_dense(m.avg_dense, w[0], w[1]); load_dense(m.max_dense, w[2], w[3]); load_dense(m.channel_dense, w[4], w[5])
    m.spatial.weight.data, m.spatial.bias.data = t(w[6].transpose(3, 2, 0, 1)), t(w[7])
    out = m(t(REF["cbam_in"].transpose(0, 3, 1, 2))).numpy().transpose(0, 2, 3, 1)
    assert np.abs(out - REF["cbam_out"]).max() < TOL


@torch.no_grad()
def test_capsule_fusion_matches_tf():
    f = CapsuleFusionBlock([256, 768]).eval()
    w = [REF[f"fus_w{i}"] for i in range(16)]
    load_dense(f.mlps[0][0], w[0], w[1]); load_dense(f.mlps[0][3], w[2], w[3])
    load_dense(f.mlps[1][0], w[6], w[7]); load_dense(f.mlps[1][3], w[8], w[9])
    for bn, g, b, mean, var in ((f.bns[0], w[4], w[5], w[12], w[13]), (f.bns[1], w[10], w[11], w[14], w[15])):
        bn.weight.data, bn.bias.data, bn.running_mean.data, bn.running_var.data = t(g), t(b), t(mean), t(var)
    out = f([t(REF["fus_in1"]), t(REF["fus_in2"])]).numpy()
    assert np.abs(out - REF["fus_out"]).max() < TOL


@torch.no_grad()
def test_refinement_matches_tf():
    r = TransformerRefinementBlock().eval()
    w = [REF[f"trb_w{i}"] for i in range(len(REF["trb_names"]))]
    load_dense(r.proj, w[0], w[1])
    r.pos_embedding.data = t(w[2])
    for li, layer in enumerate(r.layers):
        o = 3 + 16 * li  # 16 tensors per encoder block
        for k, lin in enumerate((layer.mha.q, layer.mha.k, layer.mha.v)):
            kernel, bias = w[o + 2 * k], w[o + 2 * k + 1]  # (D, H, K), (H, K)
            load_dense(lin, kernel.reshape(kernel.shape[0], -1), bias.reshape(-1))
        load_dense(layer.mha.out, w[o + 6].reshape(-1, w[o + 6].shape[-1]), w[o + 7])  # (H, K, D)
        load_dense(layer.ffn[0], w[o + 8], w[o + 9]); load_dense(layer.ffn[2], w[o + 10], w[o + 11])
        layer.norm1.weight.data, layer.norm1.bias.data = t(w[o + 12]), t(w[o + 13])
        layer.norm2.weight.data, layer.norm2.bias.data = t(w[o + 14]), t(w[o + 15])
    assert np.abs(r(t(REF["trb_in"])).numpy() - REF["trb_out"]).max() < TOL


@torch.no_grad()
def test_whole_model_shapes_and_phases():
    m = HybridPillNet(pretrained=False, num_shapes=13, num_colours=12, num_measured=5).eval()
    out = m(torch.rand(2, 3, 224, 224), measured=torch.rand(2, 5))
    assert out["embedding"].shape == (2, 256)
    assert torch.allclose(out["embedding"].norm(dim=1), torch.ones(2), atol=1e-5)
    assert out["shape_logits"].shape == (2, 13) and out["colour_logits"].shape == (2, 12)
    m.set_phase(1)
    assert not any(p.requires_grad for p in m.vit.parameters())
    assert not any(p.requires_grad for p in m.vgg[:17].parameters())
    assert all(p.requires_grad for p in m.vgg[17:].parameters())
    m.set_phase(2)
    assert all(p.requires_grad for p in m.parameters())
