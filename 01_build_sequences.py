#!/usr/bin/env python3
"""Step 1: build the mRNA-sense reference sequence and locate each oligo target.

    python 01_build_sequences.py --config config.json [--mode hexamer|peaks]
    python 01_build_sequences.py --config config.json --dry-run

Fetches the ROI plus a flank from the reference genome, converts it to mRNA sense
(reverse complement for a '-' strand gene) and finds the first occurrence of
reverse_complement(oligo) for every oligo in oligos_csv.
The flank differs per scoring mode (defaults: hexamer 300 nt, peaks 500 nt), so one set of
files is written per mode.
Output: sequences/<mode>/ref_mrna.txt, meta.json, oligo_targets.tsv
"""
import json
import sys

from common import (base_parser, load_config, locus_from_cfg, Ctx, need, read_oligos,
                    fetch_reference, revcomp, Coords, classify_region, MODES)

DEFAULT_FLANK = {"hexamer": 300, "peaks": 500}


def build(ctx, mode, locus, oligos):
    flank = int(ctx.cfg.get(mode, {}).get("flank", DEFAULT_FLANK[mode]))
    rs, re_ = locus["roi"]
    fs, fe = max(0, rs - flank), re_ + flank
    print("[%s] fetch %s:%d-%d (ROI +/- %d nt, 0-based half-open), strand %s"
          % (mode, locus["chrom"], fs, fe, flank, locus["strand"]))
    if ctx.dry:
        return
    plus = fetch_reference(ctx.cfg, locus["chrom"], fs, fe)
    if len(plus) != fe - fs:
        sys.exit("[ERROR] fetched %d bp, expected %d" % (len(plus), fe - fs))
    mrna = revcomp(plus) if locus["strand"] == "-" else plus
    co = Coords(locus["chrom"], locus["strand"], fs, fe)

    out = ctx.path("sequences", mode)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ref_mrna.txt").write_text(mrna + "\n")
    (out / "meta.json").write_text(json.dumps(
        {"chrom": locus["chrom"], "strand": locus["strand"], "fetch_start": fs,
         "fetch_end": fe, "mrna_length": len(mrna), "flank": flank}, indent=2))

    n_found = 0
    with open(str(out / "oligo_targets.tsv"), "w") as fh:
        fh.write("oligo\ttarget_len\tfound\tmrna_pos\tgenomic_pos\tregion\n")
        for name, seq in oligos.items():
            pos = mrna.find(revcomp(seq))
            if pos == -1:
                fh.write("%s\t%d\tno\t\t\t\n" % (name, len(seq)))
                continue
            n_found += 1
            g = co.to_genomic(pos)
            fh.write("%s\t%d\tyes\t%d\t%s:%d\t%s\n" % (
                name, len(seq), pos, locus["chrom"], g, classify_region(locus["regions"], g)))
    print("[%s] wrote %s (%d bp); %d/%d oligo targets found"
          % (mode, out, len(mrna), n_found, len(oligos)))


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--mode", choices=MODES, help="default: both")
    args = ap.parse_args()
    ctx = Ctx(load_config(args.config), args.dry_run)
    locus = locus_from_cfg(ctx.cfg)
    need(ctx, ctx.cfg["oligos_csv"])
    if ctx.cfg.get("reference_fasta"):
        need(ctx, ctx.cfg["reference_fasta"])
    oligos = {} if ctx.dry else read_oligos(ctx.cfg["oligos_csv"])
    for mode in ([args.mode] if args.mode else MODES):
        build(ctx, mode, locus, oligos)


if __name__ == "__main__":
    main()
