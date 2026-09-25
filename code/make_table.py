"""Collect results/*.json into the two tables Section 2.11 needs."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "experiments", "results"))

SWEEP = ["d3w32", "d3w64", "d3w128",
         "d5w32", "d5w64", "d5w128",
         "d7w32", "d7w64", "d7w128"]
ABLAT = ["d5w64", "nowork"]
LABEL = {"d5w64": "full objective (reported)",
         "nowork": "energy constraint removed"}

COLS = [("mom", "momentum"), ("rheo", "rheology"), ("solid", "solid mass"),
        ("melt", "melt mass"), ("darcy", "Darcy")]


def load():
    out = {}
    for f in glob.glob(os.path.join(RES, "*.json")):
        r = json.load(open(f))
        out[r["tag"]] = r
    return out


def fmt(v, w=9):
    return ("%.2e" % v).rjust(w)


def main():
    R = load()
    if not R:
        print("no results yet - run ./run_experiments.sh first")
        return
    lines = []

    lines.append("Table A.  Architecture sweep.  Relative r.m.s. residuals on")
    lines.append("the common evaluation batch after an identical training")
    lines.append("budget.  W is the buoyancy-work integral, which must be")
    lines.append("positive for an admissible flow.")
    lines.append("")
    head = ("layers  neurons  params  " +
            "".join(n.rjust(11) for _, n in COLS) +
            "        W   |u|max   s")
    lines.append(head)
    lines.append("-" * len(head))
    for t in SWEEP:
        r = R.get(t)
        if not r:
            d, w = int(t[1]), int(t[3:])
            lines.append("%6d  %7d       -  (not run)" % (d, w))
            continue
        lines.append("%6d  %7d  %6d  %s  %+7.3f  %6.2f  %4.0f"
                     % (r["depth"], r["width"], r["n_params"],
                        "".join(fmt(r[k], 11) for k, _ in COLS),
                        r["work"], r["umax_mmyr"], r["seconds"]))

    lines.append("")
    lines.append("")
    lines.append("Table B.  What the energy constraint contributes.  Same")
    lines.append("architecture (5 x 64), same seed, same budget, same")
    lines.append("evaluation batch; the two runs differ only in whether the")
    lines.append("one-sided buoyancy-work penalty is present.  W is the")
    lines.append("buoyancy-work integral, which is strictly positive for any")
    lines.append("admissible steady creeping flow.")
    lines.append("")
    head = ("objective                          " +
            "".join(n.rjust(11) for _, n in COLS) +
            "        W   |u|max")
    lines.append(head)
    lines.append("-" * len(head))
    for t in ABLAT:
        r = R.get(t)
        if not r:
            lines.append("%-34s  (not run)" % LABEL[t])
            continue
        lines.append("%-34s%s  %+7.3f  %6.2f"
                     % (LABEL[t],
                        "".join(fmt(r[k], 11) for k, _ in COLS),
                        r["work"], r["umax_mmyr"]))

    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(RES, "tables.txt"), "w").write(txt + "\n")
    print("\nwritten to results/tables.txt")

    missing = [t for t in set(SWEEP + ABLAT) if t not in R]
    if missing:
        print("still to run:", " ".join(sorted(missing)))


if __name__ == "__main__":
    main()
