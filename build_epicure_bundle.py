#!/usr/bin/env python3
"""
Build a browser-friendly Epicure bundle from HuggingFace.
Downloads embeddings, vocab, modes, and supervised poles,
converts to compact JSON/array formats for in-browser use.
"""

import json
import struct
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    import numpy as np
except ImportError:
    print("Install: pip install huggingface_hub safetensors numpy")
    sys.exit(1)

REPO = "Kaikaku/epicure-core"
OUT = Path("epicure_data")

def main():
    OUT.mkdir(exist_ok=True)
    print("Downloading from HuggingFace...")

    # Download all needed files
    files = {
        "embeddings.safetensors": "embeddings.safetensors",
        "vocab.json": "vocab.json",
        "itos.json": "itos.json",
        "modes.json": "modes.json",
        "supervised_poles.json": "supervised_poles.json",
        "config.json": "config.json",
    }

    for remote, local in files.items():
        path = hf_hub_download(repo_id=REPO, filename=remote)
        print(f"  Downloaded: {remote} ({Path(path).stat().st_size} bytes)")

    # Load embeddings
    E = load_file(path.replace(remote, "embeddings.safetensors"))["embeddings"]
    # Already unit-normalized by the package, but let's normalize again
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.maximum(norms, 1e-9)
    E = E.astype(np.float32)

    # Load metadata
    with open(path.replace(remote, "vocab.json")) as f:
        vocab = json.load(f)
    with open(path.replace(remote, "modes.json")) as f:
        modes_raw = json.load(f)
    with open(path.replace(remote, "supervised_poles.json")) as f:
        sup_raw = json.load(f)

    # Convert embeddings to flat float32 array (binary dump)
    # 1790 * 300 * 4 = 2,148,000 bytes
    emb_bytes = E.tobytes()
    with open(OUT / "embeddings.bin", "wb") as f:
        f.write(emb_bytes)

    # Use itos.json directly — it's a {str(int): name} dict with string keys
    itos_path = hf_hub_download(repo_id=REPO, filename="itos.json")
    with open(itos_path) as f:
        itos_raw = json.load(f)
    itos = [itos_raw[str(i)] for i in range(len(itos_raw))]

    with open(OUT / "itos.json", "w") as f:
        json.dump(itos, f)

    # Write vocab lookup (name -> index) for fast lookups
    with open(OUT / "vocab.json", "w") as f:
        json.dump(vocab, f)

    # Write modes (compact: only what we need for frontend)
    modes_out = []
    for m in modes_raw:
        modes_out.append({
            "id": m["mode_id"],
            "kind": m["kind"],
            "label": m["label"],
            "members": m["members"],
            "pole": [float(x) for x in m["pole"]],
        })
    with open(OUT / "modes.json", "w") as f:
        json.dump(modes_out, f)

    # Write supervised poles (compact)
    sup_out = {}
    for k, v in sup_raw.items():
        sup_out[k] = [float(x) for x in v]
    with open(OUT / "supervised_poles.json", "w") as f:
        json.dump(sup_out, f)

    print(f"\nDone! Files in {OUT}/:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")

    print(f"\nTotal: {sum(f.stat().st_size for f in OUT.iterdir()):,} bytes")

if __name__ == "__main__":
    main()
