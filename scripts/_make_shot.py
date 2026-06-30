"""Render the UI to a static page (with a real prediction) for screenshotting."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qsensor.inference import Predictor
from qsensor.signals import generate_one

p = Predictor()
best = None
for seed in range(40):                      # pick the clearest fault example
    sig = generate_one(fault=True, seed=seed)
    r = p.predict(sig.tolist(), model="vqc")
    if r["label"] == 1 and (best is None or r["confidence"] > best[1]["confidence"]):
        best = (sig, r)
sig, res = best
print("chosen prediction:", res)

html = open("qsensor/static/index.html", encoding="utf-8").read()
inject = (
    "  const SIG = " + json.dumps([round(float(x), 3) for x in sig]) + ";\n"
    "  sigEl.value = SIG.map(x=>x.toFixed(3)).join(', ');\n"
    "  drawSpark(SIG);\n"
    "  showResult(" + json.dumps(res) + ");\n"
)
html = html.replace("  gen(false);   // load a starter signal", inject)
open("_shot.html", "w", encoding="utf-8").write(html)
print("wrote _shot.html")
