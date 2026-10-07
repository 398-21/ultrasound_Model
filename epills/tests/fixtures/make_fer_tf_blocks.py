"""
Makes fer_tf_blocks.npz, the reference for tests/test_hybrid_parity.py: the FER notebook's own TensorFlow blocks
(CBAM, capsule fusion, refinement), taken from the notebook's cells, with random weights, run on fixed inputs.

Run in the FER TensorFlow environment (WSL, ~/venvs/fer), on the CPU so TF32 rounding doesn't blur the comparison:
    CUDA_VISIBLE_DEVICES= ~/venvs/fer/bin/python make_fer_tf_blocks.py
then compress: np.savez_compressed("fer_tf_blocks.npz", **np.load("fer_tf_blocks_raw.npz")).
"""
import json
import os
import re

import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D,
                                     Input, Lambda, Multiply, Reshape)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fer_tf_blocks_raw.npz")
NOTEBOOK = os.environ.get("FER_NOTEBOOK", os.path.join(HERE, "..", "..", "..", "FER", "exp-tuning.ipynb"))
nb = json.load(open(NOTEBOOK, encoding="utf-8"))
src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
ns = dict(tf=tf, BatchNormalization=BatchNormalization, Concatenate=Concatenate, Conv2D=Conv2D, Dense=Dense,
          Dropout=Dropout, GlobalAveragePooling2D=GlobalAveragePooling2D, Lambda=Lambda, Multiply=Multiply,
          Reshape=Reshape)
exec(next(s for s in src if "def squash" in s), ns)
exec(next(s for s in src if "class PositionalEmbedding" in s), ns)
cell = next(s for s in src if "def attention_block" in s)
exec(re.search(r"def attention_block.*?return refined_output\n", cell, re.S).group(0), ns)

rng = np.random.default_rng(0)
save = {}
# CBAM
x = Input((7, 7, 512))
cbam = tf.keras.Model(x, ns["attention_block"](x))
for w in cbam.weights:
    w.assign(rng.normal(0, 0.05, w.shape).astype("float32"))
xin = rng.normal(0, 1, (3, 7, 7, 512)).astype("float32")
save["cbam_in"], save["cbam_out"] = xin, cbam(xin).numpy()
for i, w in enumerate(cbam.weights):
    save[f"cbam_w{i}"] = w.numpy()
# fusion
fusion = ns["AdvancedCapsuleFusionBlock"](capsule_dim=256, num_iterations=2, dropout_rate=0.4, num_inputs=2)
f1, f2 = rng.normal(0, 1, (3, 256)).astype("float32"), rng.normal(0, 1, (3, 768)).astype("float32")
fusion([f1, f2])
for w in fusion.weights:
    w.assign((rng.uniform(0.5, 1.5, w.shape) if "variance" in w.name else rng.normal(0, 0.05, w.shape)).astype("float32"))
save["fus_in1"], save["fus_in2"], save["fus_out"] = f1, f2, fusion([f1, f2], training=False).numpy()
for i, w in enumerate(fusion.weights):
    save[f"fus_w{i}"] = w.numpy()
save["fus_names"] = np.array([w.name for w in fusion.weights])
# refinement
trb = ns["AdvancedTransformerRefinementBlock"](num_tokens=4, token_dim=64, num_heads=4, ff_dim=128, num_layers=2, dropout_rate=0.4)
t_in = rng.normal(0, 1, (3, 256)).astype("float32")
trb(t_in)
for w in trb.weights:
    w.assign(rng.normal(0, 0.05, w.shape).astype("float32"))
save["trb_in"], save["trb_out"] = t_in, trb(t_in, training=False).numpy()
for i, w in enumerate(trb.weights):
    save[f"trb_w{i}"] = w.numpy()
save["trb_names"] = np.array([w.name for w in trb.weights])
save["cbam_names"] = np.array([w.name for w in cbam.weights])
np.savez(OUT, **save)
print("saved", len(save))
