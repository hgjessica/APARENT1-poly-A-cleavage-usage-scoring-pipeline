#!/usr/bin/env python3
"""Step 2: APARENT1 poly(A) scoring of the reference vs. oligo-masked sequence.

    python 02_score_aparent1.py --config config.json hexamer --mask n
    python 02_score_aparent1.py --config config.json hexamer --mask scramble
    python 02_score_aparent1.py --config config.json peaks
    python 02_score_aparent1.py --config config.json submit hexamer --mask n
    python 02_score_aparent1.py --config config.json --dry-run hexamer

hexamer : scan the ROI for canonical poly(A) hexamers inside locus.score_regions, place each
          hexamer at position 70 of a 205 nt window, score reference vs. masked window.
          Reports isoform probability, peak cleavage probability and their product
          ("composite").
peaks   : detect poly(A) peaks on the reference (strided scan of the whole padded sequence),
          then score every peak as log-odds on reference and masked sequence.
Masking : the oligo target (reverse complement of the oligo, mRNA sense) is replaced by N
          (all-zero one-hot) or by a composition-preserving shuffle (--mask scramble).
Needs step 01 first. Run on CPU; use `submit` (sbatch) rather than a blocking srun.
Output: scores/hexamer_scores_<mask>.csv  or  scores/peak_scores_Nmask_all.csv (+ _score_regions.csv)
"""
import argparse
import sys

import numpy as np
import pandas as pd

from common import (base_parser, load_config, locus_from_cfg, Ctx, need, read_oligos,
                    load_sequences, load_aparent, Coords, classify_region, mask_target,
                    submit_job, APARENT_SEQ_LEN, DEFAULT_HEXAMERS)


def find_hexamers(seq, hexamers):
    """All (overlapping) occurrences of each hexamer: [(hexamer, index)]."""
    hits = []
    for hx in hexamers:
        i = seq.find(hx)
        while i != -1:
            hits.append((hx, i))
            i = seq.find(hx, i + 1)
    return hits


def predict_window(model, encoder, window):
    """-> (isoform probability, peak cleavage probability, composite)."""
    iso, cut = model.predict(x=encoder([window]))
    score = float(iso[0, 0])
    peak = float(np.max(cut[0][:APARENT_SEQ_LEN]))     # index 205 is the no-cut class
    return score, peak, score * peak


# ------------------------------------------------------------------ hexamer mode
def step_hexamer(ctx, args):
    cfg = ctx.cfg
    hcfg = cfg.get("hexamer", {})
    hexamers = [h.upper() for h in hcfg.get("hexamers", DEFAULT_HEXAMERS)]
    hex_pos = int(hcfg.get("hexamer_pos", 70))        # hexamer start inside the 205 nt window
    seed = int(cfg.get("masking", {}).get("scramble_seed", 42))
    locus = locus_from_cfg(cfg)
    print("hexamer scan: %d hexamers, window %d nt with hexamer at index %d, mask=%s"
          % (len(hexamers), APARENT_SEQ_LEN, hex_pos, args.mask))
    if ctx.dry:
        print("[dry-run] would load sequences/hexamer, the model, and write scores/hexamer_scores_%s.csv"
              % args.mask)
        return
    meta, mrna = load_sequences(ctx, "hexamer")
    co = Coords(meta["chrom"], meta["strand"], meta["fetch_start"], meta["fetch_end"])
    oligos = read_oligos(cfg["oligos_csv"])

    a, b = co.roi_slice(locus["roi"])
    sites = []
    for hx, i in find_hexamers(mrna[a:b], hexamers):
        m = i + a
        g = co.to_genomic(m)
        reg = classify_region(locus["regions"], g)
        if reg in locus["score_regions"]:
            sites.append((hx, m, g, reg))
    print("%d hexamer sites in %s" % (len(sites), sorted(locus["score_regions"])))
    if not sites:
        sys.exit("[ERROR] no poly(A) hexamers in the scored regions")

    model, pred = load_aparent(cfg)
    encoder = pred.get_aparent_encoder()
    masked = {}
    for name, seq in oligos.items():
        mm, pos = mask_target(mrna, seq, args.mask, seed)
        if pos == -1:
            print("[skip] %s: target not found in the fetched sequence" % name)
        else:
            masked[name] = mm

    rows = []
    for hx, m, g, reg in sites:
        ws = m - hex_pos
        we = ws + APARENT_SEQ_LEN
        if ws < 0 or we > len(mrna):
            print("[skip] %s at %d: window out of bounds (increase hexamer.flank)" % (hx, m))
            continue
        wt_window = mrna[ws:we]
        wt_iso, wt_cut, wt_comp = predict_window(model, encoder, wt_window)
        for name, mm in masked.items():
            mwin = mm[ws:we]
            if mwin == wt_window:          # oligo target does not touch this window
                continue
            mk_iso, mk_cut, mk_comp = predict_window(model, encoder, mwin)
            d_iso, d_comp = mk_iso - wt_iso, mk_comp - wt_comp
            rows.append({
                "oligo": name, "polyA_hexamer": hx,
                "hexamer_genomic_pos": "%s:%d" % (meta["chrom"], g), "hexamer_region": reg,
                "WT_iso_prob": round(wt_iso, 6), "masked_iso_prob": round(mk_iso, 6),
                "WT_peak_cut_prob": round(wt_cut, 6), "masked_peak_cut_prob": round(mk_cut, 6),
                "WT_composite": round(wt_comp, 6), "masked_composite": round(mk_comp, 6),
                "delta_iso": round(d_iso, 6), "abs_delta_iso": round(abs(d_iso), 6),
                "delta_composite": round(d_comp, 6), "abs_delta_composite": round(abs(d_comp), 6),
            })
    if not rows:
        sys.exit("[ERROR] no oligo / hexamer window overlaps found")
    df = pd.DataFrame(rows).sort_values("abs_delta_composite", ascending=False)
    out = ctx.path("scores")
    out.mkdir(parents=True, exist_ok=True)
    f = out / ("hexamer_scores_%s.csv" % args.mask)
    df.to_csv(str(f), index=False)
    print(df.to_string(index=False))
    print("Saved %s" % f)


# ------------------------------------------------------------------ peaks mode
def step_peaks(ctx, args):
    cfg = ctx.cfg
    p = cfg.get("peaks", {})
    find_kw = dict(sequence_stride=int(p.get("find_stride", 5)), conv_smoothing=True,
                   peak_min_height=float(p.get("peak_min_height", 0.05)),
                   peak_min_distance=int(p.get("peak_min_distance", 100)),
                   peak_prominence=(float(p.get("peak_prominence", 0.05)), None))
    score_kw = dict(sequence_stride=int(p.get("score_stride", 1)),
                    strided_agg_mode=p.get("score_agg", "max"),
                    iso_scoring_mode=p.get("score_iso_mode", "both"),
                    score_unit=p.get("score_unit", "log"))
    x_pad = int(p.get("x_pad", 500))
    locus = locus_from_cfg(cfg)
    print("peak mode: find %s | score %s | X padding %d nt" % (find_kw, score_kw, x_pad))
    if ctx.dry:
        print("[dry-run] would load sequences/peaks, the model, and write scores/peak_scores_Nmask_*.csv")
        return
    meta, mrna = load_sequences(ctx, "peaks")
    co = Coords(meta["chrom"], meta["strand"], meta["fetch_start"], meta["fetch_end"])
    oligos = read_oligos(cfg["oligos_csv"])
    model, pred = load_aparent(cfg)
    encoder = pred.get_aparent_encoder()
    pad = "X" * x_pad

    wt_seq = pad + mrna + pad
    peak_ixs, _ = pred.find_polya_peaks(model, encoder, wt_seq, **find_kw)
    print("%d peaks on the reference sequence" % len(peak_ixs))
    if not peak_ixs:
        sys.exit("[ERROR] no peaks found on the reference sequence")
    info = []
    for px in peak_ixs:
        g = co.to_genomic(px - x_pad)
        info.append((g, classify_region(locus["regions"], g)))
        print("  peak @ %s:%d [%s]" % (meta["chrom"], g, info[-1][1]))
    wt_scores = pred.score_polya_peaks(model, encoder, wt_seq, peak_ixs, **score_kw)

    rows = []
    for name, seq in oligos.items():
        mm, pos = mask_target(mrna, seq, "n", 0)
        if pos == -1:
            print("[skip] %s: target not found in the fetched sequence" % name)
            continue
        mg = co.to_genomic(pos)
        mreg = classify_region(locus["regions"], mg)
        scores = pred.score_polya_peaks(model, encoder, pad + mm + pad, peak_ixs, **score_kw)
        for i in range(len(peak_ixs)):
            d = scores[i] - wt_scores[i]
            rows.append({
                "oligo": name, "oligo_target_pos": "%s:%d" % (meta["chrom"], mg),
                "oligo_target_region": mreg,
                "peak_genomic_pos": "%s:%d" % (meta["chrom"], info[i][0]),
                "peak_region": info[i][1],
                "WT_log_odds": round(wt_scores[i], 4), "masked_log_odds": round(scores[i], 4),
                "delta": round(d, 4), "abs_delta": round(abs(d), 4),
                "in_score_regions": info[i][1] in locus["score_regions"],
            })
    if not rows:
        sys.exit("[ERROR] no results produced")
    df = pd.DataFrame(rows).sort_values("abs_delta", ascending=False)
    out = ctx.path("scores")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out / "peak_scores_Nmask_all.csv"), index=False)
    roi = df[df["in_score_regions"]]
    roi.to_csv(str(out / "peak_scores_Nmask_score_regions.csv"), index=False)
    print(roi.to_string(index=False) if len(roi) else "(no peaks inside score_regions)")
    print("Saved %s" % out)


def main():
    ap = base_parser(__doc__)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True
    h = sub.add_parser("hexamer", help="canonical-hexamer windows, plain or scrambled/N mask")
    h.add_argument("--mask", choices=("n", "scramble"), default="n")
    h.set_defaults(step_name="hexamer")
    pk = sub.add_parser("peaks", help="peak detection + log-odds scoring (N-masking)")
    pk.set_defaults(step_name="peaks")
    s = sub.add_parser("submit", help="sbatch this step as a CPU job")
    s.add_argument("step_name", choices=("hexamer", "peaks"), metavar="step")
    s.add_argument("step_args", nargs=argparse.REMAINDER, help="extra args, e.g. --mask scramble")
    args = ap.parse_args()
    ctx = Ctx(load_config(args.config), args.dry_run)
    need(ctx, ctx.cfg["oligos_csv"])
    if args.cmd == "submit":
        return submit_job(ctx, args, __file__)
    {"hexamer": step_hexamer, "peaks": step_peaks}[args.step_name](ctx, args)


if __name__ == "__main__":
    main()
