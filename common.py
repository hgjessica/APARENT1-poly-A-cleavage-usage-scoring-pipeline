"""Shared helpers for the APARENT1 poly(A) scoring steps.

Python 3.6 compatible (the original APARENT1 environment pins Python 3.6 / TF 1.13).
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

APARENT_SEQ_LEN = 205          # APARENT1 input length (fixed by the model)
DEFAULT_MODEL = "aparent_large_lessdropout_all_libs_no_sampleweights.h5"
DEFAULT_HEXAMERS = ["AATAAA", "ATTAAA", "TATAAA", "AGTAAA", "AATACA",
                    "AATATA", "AATAGA", "ACTAAA", "AATGAA", "CATAAA"]
MODES = ("hexamer", "peaks")


# ---------------------------------------------------------------- context / config
class Ctx(object):
    def __init__(self, cfg, dry_run):
        self.cfg = cfg
        self.dry = dry_run
        self.work = Path(cfg["work_dir"])

    def path(self, *parts):
        return self.work.joinpath(*parts)


def base_parser(doc):
    parts = doc.strip().split("\n\n")
    ap = argparse.ArgumentParser(description=parts[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n\n".join(parts[1:]))
    ap.add_argument("--config", required=True, help="JSON config (see config.example.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, do not read the model or write results")
    return ap


def parse_interval(text, what):
    """'chrN:START-END' -> (chrom, start, end). Refuses un-filled placeholders."""
    m = re.match(r"^([\w.]+):(\d[\d,_]*)-(\d[\d,_]*)$", str(text).strip())
    if not m:
        sys.exit("[ERROR] %s must look like chr1:1000-2000 (got %r; replace the "
                 "START/END placeholders)" % (what, text))
    s, e = (int(x.replace(",", "").replace("_", "")) for x in (m.group(2), m.group(3)))
    if e <= s:
        sys.exit("[ERROR] %s: END must be larger than START (%r)" % (what, text))
    return m.group(1), s, e


def load_config(path):
    with open(path) as fh:
        cfg = json.load(fh)
    for key in ("work_dir", "locus", "oligos_csv"):
        if key not in cfg:
            sys.exit("[ERROR] config is missing required key: %s" % key)
    loc = cfg["locus"]
    for key in ("roi", "strand", "regions", "score_regions"):
        if key not in loc:
            sys.exit("[ERROR] config.locus is missing required key: %s" % key)
    if loc["strand"] not in ("+", "-"):
        sys.exit("[ERROR] locus.strand must be '+' or '-'")
    return cfg


def locus_from_cfg(cfg):
    """Validated locus: chrom, strand, roi (0-based half-open), regions [(name, s, e)]."""
    loc = cfg["locus"]
    chrom, rs, re_ = parse_interval(loc["roi"], "locus.roi")
    regions = []
    for r in loc["regions"]:
        c, s, e = parse_interval(r["interval"], "region %s" % r.get("name", "?"))
        if c != chrom:
            sys.exit("[ERROR] region %s is on %s but the ROI is on %s" % (r["name"], c, chrom))
        regions.append((r["name"], s, e))
    names = [r[0] for r in regions]
    for n in loc["score_regions"]:
        if n not in names:
            sys.exit("[ERROR] score_regions entry %r is not defined in regions" % n)
    return {"chrom": chrom, "strand": loc["strand"], "roi": (rs, re_),
            "regions": regions, "score_regions": set(loc["score_regions"])}


def need(ctx, *files):
    for f in files:
        if not ctx.dry and not Path(f).exists():
            sys.exit("[ERROR] missing: %s" % f)


def run(ctx, cmd):
    print("$ " + " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if ctx.dry:
        return 0
    return subprocess.run([str(c) for c in cmd]).returncode


# ---------------------------------------------------------------- sequence utilities
_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def revcomp(seq):
    return seq.translate(_COMP)[::-1]


def read_oligos(path):
    """CSV/TSV with columns name,sequence (header required; case-insensitive).

    Sequences are the oligo 5'->3' (antisense to the mRNA). Returns {name: SEQ}.
    """
    import pandas as pd
    sep = "\t" if str(path).endswith(".tsv") else ","
    df = pd.read_csv(str(path), sep=sep, dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    if "name" not in df.columns or "sequence" not in df.columns:
        sys.exit("[ERROR] %s needs header columns: name,sequence" % path)
    df = df.dropna(subset=["name", "sequence"])
    return dict(zip(df["name"].str.strip(), df["sequence"].str.strip().str.upper()))


def ensure_fai(fasta):
    """Return {chrom: (length, offset, linebases, linewidth)} from <fasta>.fai.

    Builds the index in pure Python if it is missing (no samtools needed).
    """
    fai = fasta + ".fai"
    if not os.path.exists(fai):
        print("[INFO] building %s" % fai)
        rows, name, length, offset, lb, lw = [], None, 0, 0, 0, 0
        with open(fasta, "rb") as fh:
            pos = 0
            for line in iter(fh.readline, b""):
                if line.startswith(b">"):
                    if name is not None:
                        rows.append((name, length, offset, lb, lw))
                    name, length, lb, lw = line[1:].split()[0].decode(), 0, 0, 0
                    offset = pos + len(line)
                else:
                    stripped = len(line.rstrip(b"\r\n"))
                    if lb == 0:
                        lb, lw = stripped, len(line)
                    length += stripped
                pos += len(line)
            if name is not None:
                rows.append((name, length, offset, lb, lw))
        with open(fai, "w") as out:
            for r in rows:
                out.write("%s\t%d\t%d\t%d\t%d\n" % r)
    idx = {}
    with open(fai) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            idx[f[0]] = tuple(int(x) for x in f[1:5])
    return idx


def fetch_fasta(fasta, chrom, start, end):
    """Plus-strand sequence of 0-based half-open [start, end), upper-case."""
    idx = ensure_fai(fasta)
    if chrom not in idx:
        sys.exit("[ERROR] %s not in %s (check chromosome naming, e.g. chr1 vs 1)" % (chrom, fasta))
    length, offset, lb, lw = idx[chrom]
    if start < 0 or end > length:
        sys.exit("[ERROR] %s:%d-%d is outside the chromosome (length %d); shrink the flank "
                 "or the ROI" % (chrom, start, end, length))
    first, last = start // lb, (end - 1) // lb
    with open(fasta, "rb") as fh:
        fh.seek(offset + first * lw)
        raw = fh.read((last - first + 1) * lw)
    seq = raw.decode().replace("\n", "").replace("\r", "")
    s0 = start - first * lb
    return seq[s0:s0 + (end - start)].upper()


def fetch_ucsc(genome, chrom, start, end):
    """Fallback used by the original scripts: UCSC REST API (needs internet)."""
    import requests
    url = ("https://api.genome.ucsc.edu/getData/sequence"
           "?genome=%s;chrom=%s;start=%d;end=%d" % (genome, chrom, start, end))
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    return resp.json()["dna"].upper()


def fetch_reference(cfg, chrom, start, end):
    if cfg.get("reference_fasta"):
        return fetch_fasta(cfg["reference_fasta"], chrom, start, end)
    if cfg.get("ucsc_genome"):
        return fetch_ucsc(cfg["ucsc_genome"], chrom, start, end)
    sys.exit("[ERROR] set reference_fasta (or ucsc_genome) in the config")


# ---------------------------------------------------------------- coordinates
class Coords(object):
    """Map between mRNA-sense index m and 0-based genomic position.

    The mRNA-sense sequence is the fetched plus-strand window for a '+' gene and its
    reverse complement for a '-' gene. All genomic positions are 0-based.
    """
    def __init__(self, chrom, strand, fetch_start, fetch_end):
        self.chrom, self.strand = chrom, strand
        self.fs, self.fe = fetch_start, fetch_end

    def to_genomic(self, m):
        return self.fe - 1 - m if self.strand == "-" else self.fs + m

    def roi_slice(self, roi):
        """mRNA-sense [a, b) covering genomic ROI [start, end)."""
        s, e = roi
        if self.strand == "-":
            return self.fe - e, self.fe - s
        return s - self.fs, e - self.fs


def classify_region(regions, gpos):
    """First region (name, start, end inclusive) containing gpos, else 'other'."""
    for name, s, e in regions:
        if s <= gpos <= e:
            return name
    return "other"


def mask_target(mrna, oligo, how, seed):
    """Mask the first occurrence of revcomp(oligo) in the mRNA-sense sequence.

    how='n': replace with N (not in APARENT1's encode map -> all-zero one-hot).
    how='scramble': composition-preserving shuffle with RandomState(seed).
    Returns (masked_seq, pos) with pos=-1 (sequence unchanged) if not found.
    """
    target = revcomp(oligo)
    pos = mrna.find(target)
    if pos == -1:
        return mrna, -1
    if how == "n":
        repl = "N" * len(target)
    else:
        arr = list(target)
        np.random.RandomState(seed).shuffle(arr)
        repl = "".join(arr)
    return mrna[:pos] + repl + mrna[pos + len(target):], pos


# ---------------------------------------------------------------- step-01 products
def seq_dir(ctx, mode):
    return ctx.path("sequences", mode)


def load_sequences(ctx, mode):
    d = seq_dir(ctx, mode)
    need(ctx, d / "meta.json", d / "ref_mrna.txt")
    meta = json.loads((d / "meta.json").read_text())
    mrna = (d / "ref_mrna.txt").read_text().strip()
    return meta, mrna


# ---------------------------------------------------------------- model
def load_aparent(cfg):
    """Import the (separately installed) aparent package and load the APARENT1 weights."""
    adir = cfg.get("aparent_dir")
    if adir:
        sys.path.insert(0, str(Path(adir)))
    model_h5 = cfg.get("model_h5") or str(Path(adir or ".") / "saved_models" / DEFAULT_MODEL)
    if not Path(model_h5).exists():
        sys.exit("[ERROR] APARENT1 weights not found: %s" % model_h5)
    from keras.models import load_model
    import aparent.predictor as pred
    return load_model(model_h5), pred


# ---------------------------------------------------------------- SLURM
def submit_job(ctx, args, script):
    """Write and sbatch one CPU job that re-runs this step. Never uses srun."""
    sl = ctx.cfg.get("slurm", {})
    extra = " ".join(shlex.quote(a) for a in args.step_args)
    log_dir = ctx.path("slurm_logs")
    text = """#!/bin/bash
#SBATCH -p {part}
#SBATCH -N 1
#SBATCH -n {n}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH --export=ALL
#SBATCH -J aparent1_{step}
#SBATCH -o {logs}/{step}_%j.out
#SBATCH -e {logs}/{step}_%j.err
{gres}
source {conda_sh}
conda activate {env}
python {script} --config {cfg} {step} {extra}
""".format(part=sl.get("partition", "PARTITION_NAME"), n=sl.get("cpus", 4),
           mem=sl.get("mem", "16G"), time=sl.get("time", "04:00:00"),
           step=args.step_name, logs=log_dir,
           gres=("#SBATCH --gres=gpu:1" if sl.get("gpu") else ""),
           conda_sh=sl.get("conda_sh", "Path_to_conda.sh"), env=sl.get("conda_env", "aparent1"),
           script=Path(script).resolve(), cfg=Path(args.config).resolve(), extra=extra)
    print(text)
    if ctx.dry:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    f = ctx.path("slurm", "%s.sbatch" % args.step_name)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    subprocess.run(["sbatch", str(f)])
