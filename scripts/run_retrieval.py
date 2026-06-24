#!/usr/bin/env python
"""Score every query against the candidate pool for each retrieval baseline.

Baselines
  random          : random per-query ranking (chance floor)
  oracle          : transcriptomic marker / composition score (upper bound)
  biomedclip      : BiomedCLIP zero-shot text<->H&E patch cosine (medium scale)
  biomedclip_nbhd : same, neighborhood-scale patches (ablation)
  neighborhood    : text vs mean of spatial-neighbor patch embeddings (Baseline E)
  linear_probe    : frozen patch embedding + 5-fold out-of-fold ridge (Baseline D)

Writes results/retrieval/scores_<baseline>.npy (n_queries x n_pool) and a copy
of the query table. Image baselines are silently skipped if embeddings are
missing, so the non-image baselines always run.
"""
import os

import numpy as np
import pandas as pd

from _bootstrap import abspath, common_parser, load_config, results_dir
from tiletalk import data as D
from tiletalk import index as IDX
from tiletalk import queries as Q
from tiletalk import features as F
from tiletalk import probe as PB

# niche label -> oracle composition score over a pool DataFrame
NICHE_ORACLE = {
    "bcell_rich":  lambda c: c["frac_B_cell"].values,
    "tcell_rich":  lambda c: c["frac_T_cell"].values,
    "myeloid_rich": lambda c: c["frac_Myeloid"].values,
    "vascular":    lambda c: c["frac_Endothelial"].values,
    "immune_rich": lambda c: (c["frac_T_cell"] + c["frac_B_cell"] + c["frac_Myeloid"]).values,
    "tumor_immune_interface":
        lambda c: (c["frac_Epithelial"] * (c["frac_T_cell"] + c["frac_Myeloid"])).values,
}


def oracle_scores(qs, pool, expr, gene_names):
    """Transcriptomic upper-bound score vector per query (n_q x n_pool)."""
    gi = {g: i for i, g in enumerate(map(str, gene_names))}
    out = np.zeros((len(qs), len(pool)), dtype=np.float32)
    for r, q in enumerate(qs.itertuples()):
        f, lab = q.target_field, q.target_label
        if f == "cell_type":
            out[r] = pool[f"score_{lab}"].values
        elif f == "cell_state":
            out[r] = pool["proliferation_score"].values
        elif f == "gene":
            out[r] = expr[:, gi[lab]] if lab in gi else 0.0
        elif f == "niche":
            out[r] = NICHE_ORACLE[lab](pool)
    return out


def random_scores(qs, pool, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((len(qs), len(pool))).astype(np.float32)


def clip_scores(text_emb, img_emb):
    return IDX.cosine_scores(text_emb, img_emb).astype(np.float32)


def neighbor_average(emb, coords, k):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords)
    _, idx = nn.kneighbors(coords)            # includes self
    return emb[idx].mean(axis=1).astype(np.float32)


def linear_probe_scores(qs, pool, expr, gene_names, img_emb, nfolds=5, seed=0):
    """5-fold out-of-fold ridge from frozen patch embeddings to query relevance."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    n = len(pool)
    out = np.zeros((len(qs), n), dtype=np.float32)
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    rel_cache = [Q.relevance_mask(q, pool, expr, gene_names).astype(np.float32)
                 for _, q in qs.iterrows()]
    for r, y in enumerate(rel_cache):
        if y.sum() == 0:
            continue
        for tr, te in kf.split(img_emb):
            m = Ridge(alpha=1.0).fit(img_emb[tr], y[tr])
            out[r, te] = m.predict(img_emb[te])
    return out


# Pathology prompt templates for fair zero-shot evaluation (CLIP-style ensembling).
PROMPT_TEMPLATES = [
    "{}", "an H&E image of {}", "a histopathology image showing {}",
    "a pathology slide with {}", "hematoxylin and eosin stain of {}",
    "a microscopy image of {} in breast tissue",
]


def prompt_ensemble_text(encoder, queries):
    """Average the L2-normalized text embeddings over pathology prompt templates."""
    embs = []
    for tmpl in PROMPT_TEMPLATES:
        texts = [tmpl.format(q) for q in queries]
        embs.append(IDX.l2_normalize(encoder.encode_text(texts)))
    return IDX.l2_normalize(np.mean(embs, axis=0))


def openvocab_oof(feat, qs, pool, lineage_text_emb, lineage_names, query_text_emb,
                  nfolds=5, seed=0, device="cuda"):
    """Open-vocabulary head: learn image-feature -> text-embedding projection on
    (cell, cell-type-name) pairs, then score any query by cosine in text space.
    Generalizes to unseen queries without per-query training (out-of-fold)."""
    import torch
    from sklearn.model_selection import KFold
    dev = device if torch.cuda.is_available() else "cpu"
    name_to_row = {n: i for i, n in enumerate(lineage_names)}
    # target text embedding per cell = embedding of its cell-type name (unknown->mean)
    tgt = np.zeros((len(pool), query_text_emb.shape[1]), dtype=np.float32)
    have = np.zeros(len(pool), dtype=bool)
    for i, ct in enumerate(pool["cell_type"].values):
        if ct in name_to_row:
            tgt[i] = lineage_text_emb[name_to_row[ct]]; have[i] = True
    Xt = torch.tensor(feat, dtype=torch.float32, device=dev)
    Yt = torch.tensor(tgt, dtype=torch.float32, device=dev)
    proj = np.zeros_like(tgt)
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    for tr, te in kf.split(feat):
        tr = tr[have[tr]]
        mu = Xt[tr].mean(0, keepdim=True); sd = Xt[tr].std(0, keepdim=True) + 1e-6
        model = torch.nn.Linear(feat.shape[1], tgt.shape[1]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-3)
        tri = torch.tensor(tr, device=dev)
        for _ in range(400):
            opt.zero_grad()
            pred = model((Xt[tri] - mu) / sd)
            pred = pred / pred.norm(dim=1, keepdim=True).clamp_min(1e-6)
            loss = (1 - (pred * Yt[tri]).sum(1)).mean()   # cosine alignment loss
            loss.backward(); opt.step()
        with torch.no_grad():
            tei = torch.tensor(te, device=dev)
            p = model((Xt[tei] - mu) / sd)
            proj[te] = (p / p.norm(dim=1, keepdim=True).clamp_min(1e-6)).cpu().numpy()
    return IDX.cosine_scores(query_text_emb, proj).astype(np.float32)


def main():
    ap = common_parser(__doc__)
    ap.add_argument("--baselines", nargs="+", default=None)
    ap.add_argument("--image-encoder", default=None,
                    help="encoder name used for emb_<name>_<scale>.npy (default: config)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = abspath(cfg["dataset"]["processed_dir"])
    outdir = results_dir(cfg, "retrieval"); os.makedirs(outdir, exist_ok=True)
    baselines = args.baselines or cfg["retrieval"]["baselines"]
    enc_name = args.image_encoder or cfg["encoders"]["image"]
    k = cfg["neighborhood"]["k"]

    # ---- load pool, expression subset, queries ----
    cells, expr_full, gene_names = D.load_processed(proc)
    pool = pd.read_parquet(os.path.join(proc, "pool_cells.parquet"))
    expr = expr_full[pool["orig_row"].values]
    qs = pd.read_csv(os.path.join(proc, "queries.csv"))
    qs.to_csv(os.path.join(outdir, "queries.csv"), index=False)
    np.save(os.path.join(outdir, "pool_orig_row.npy"), pool["orig_row"].values)
    coords = pool[["x_centroid", "y_centroid"]].values
    print(f"pool {len(pool)} cells, {len(qs)} queries, baselines={baselines}")

    # ---- generic embedding + text loaders ----
    def load_emb(enc, scale):
        p = os.path.join(proc, f"emb_{enc}_{scale}.npy")
        return np.load(p) if os.path.exists(p) else None

    _text_cache = {}

    def text_emb(enc):
        if enc not in _text_cache:
            from tiletalk import encoders as E
            ec = cfg["encoders"]
            try:
                te = E.get_encoder(enc, hub=ec["biomedclip_hub"],
                                   device=ec["device"], batch_size=ec["batch_size"])
                _text_cache[enc] = te.encode_text(list(qs["query_text"]))
                print(f"  text embeddings [{enc}]: {_text_cache[enc].shape}")
            except Exception as e:
                print(f"[warn] text encoder {enc} unavailable: {e}")
                _text_cache[enc] = None
        return _text_cache[enc]

    emb_med = load_emb(enc_name, "medium")        # primary image encoder (biomedclip)
    emb_nbhd = load_emb(enc_name, "neighborhood")

    # ---- TileTalk fused feature (multi-encoder x multi-scale + spatial context) ----
    # TileTalk uses the best available encoder set: full fusion incl. UNI2-h when
    # present (cellseek_all), else the open-encoder version (cellseek_full).
    blocks = F.load_blocks(proc, coords, k)
    _specs = F.variant_specs(blocks)
    _cs_spec = _specs.get("cellseek_all") or _specs.get("cellseek_full") or []
    cellseek_feat = F.assemble(blocks, _cs_spec)
    print(f"  TileTalk feature spec: {'cellseek_all (+UNI2-h)' if 'cellseek_all' in _specs else 'cellseek_full (open)'}")
    Y = PB.relevance_matrix(qs, pool, expr, gene_names)   # [n_pool, n_queries]
    dev = cfg["encoders"]["device"]

    def prompt_text(enc):
        if ("prompt:" + enc) not in _text_cache:
            from tiletalk import encoders as E
            ec = cfg["encoders"]
            try:
                te = E.get_encoder(enc, hub=ec["biomedclip_hub"],
                                   device=ec["device"], batch_size=ec["batch_size"])
                _text_cache["prompt:" + enc] = prompt_ensemble_text(te, list(qs["query_text"]))
            except Exception as e:
                print(f"[warn] prompt text encoder {enc} unavailable: {e}")
                _text_cache["prompt:" + enc] = None
        return _text_cache["prompt:" + enc]

    # ---- run each baseline ----
    produced = []
    for b in baselines:
        if b == "random":
            S = random_scores(qs, pool, cfg["seed"])
        elif b == "oracle":
            S = oracle_scores(qs, pool, expr, gene_names)
        elif b == "biomedclip":
            te = text_emb("biomedclip")
            if emb_med is None or te is None:
                print(f"[skip] {b}: missing medium embeddings or text encoder"); continue
            S = clip_scores(te, emb_med)
        elif b == "biomedclip_nbhd":
            te = text_emb("biomedclip")
            if emb_nbhd is None or te is None:
                print(f"[skip] {b}: missing neighborhood embeddings"); continue
            S = clip_scores(te, emb_nbhd)
        elif b == "plip":
            em = load_emb("plip", "medium"); te = text_emb("plip")
            if em is None or te is None:
                print(f"[skip] {b}: missing PLIP embeddings or text encoder"); continue
            S = clip_scores(te, em)
        elif b == "neighborhood":
            te = text_emb("biomedclip")
            if emb_med is None or te is None:
                print(f"[skip] {b}: missing embeddings"); continue
            S = clip_scores(te, neighbor_average(emb_med, coords, k))
        elif b == "biomedclip_prompt":
            te = prompt_text("biomedclip")
            if emb_med is None or te is None:
                print(f"[skip] {b}: missing embeddings or text encoder"); continue
            S = clip_scores(te, emb_med)
        elif b == "plip_prompt":
            em = load_emb("plip", "medium"); te = prompt_text("plip")
            if em is None or te is None:
                print(f"[skip] {b}: missing PLIP embeddings or text encoder"); continue
            S = clip_scores(te, em)
        elif b == "linear_probe":
            if emb_med is None:
                print(f"[skip] {b}: missing embeddings"); continue
            S = linear_probe_scores(qs, pool, expr, gene_names, emb_med, seed=cfg["seed"])
        elif b in ("cellseek", "cellseek_mlp"):
            if cellseek_feat is None:
                print(f"[skip] {b}: no frozen embeddings available"); continue
            head = "mlp" if b == "cellseek_mlp" else "logistic"
            print(f"  [{b}] fused feature dim={cellseek_feat.shape[1]}, head={head}")
            S = PB.oof_scores(cellseek_feat, Y, head=head, seed=cfg["seed"], device=dev)
        elif b == "openvocab":
            if cellseek_feat is None:
                print(f"[skip] {b}: no frozen embeddings"); continue
            from tiletalk import encoders as E
            ec = cfg["encoders"]
            te = E.get_encoder("biomedclip", hub=ec["biomedclip_hub"], device=dev,
                               batch_size=ec["batch_size"])
            lineage_names = Q.LINEAGE_TYPES
            lineage_text = IDX.l2_normalize(te.encode_text(
                [n.replace("_", " ") + "s" for n in lineage_names]))
            qtext = text_emb("biomedclip")
            S = openvocab_oof(cellseek_feat, qs, pool, lineage_text, lineage_names,
                              qtext, seed=cfg["seed"], device=dev)
        else:
            print(f"[skip] unknown baseline {b}"); continue
        np.save(os.path.join(outdir, f"scores_{b}.npy"), S)
        produced.append(b)
        print(f"  [{b}] scores {S.shape} -> scores_{b}.npy")

    print("produced baselines:", produced)


if __name__ == "__main__":
    main()
