"""Quick client demo against the running API.

Start the API first:  uvicorn qsensor.api:app
Then:                  python scripts/api_demo.py
"""
import httpx

BASE = "http://localhost:8000"


def main():
    h = httpx.get(f"{BASE}/health").json()
    print("health     :", h["status"], "| VQC acc:", h["metadata"]["metrics"]["vqc_acc"])

    for fault in (False, True):
        sample = httpx.get(f"{BASE}/sample", params={"fault": fault}).json()
        truth = sample["true_label"]
        print(f"\nsignal (true={'fault' if truth else 'normal'}):")
        for model in ("vqc", "classical"):
            r = httpx.post(f"{BASE}/predict", params={"model": model},
                           json={"signal": sample["signal"]}).json()
            print(f"  {model:9s} -> {r['prediction']:6s}  (confidence {r['confidence']})")


if __name__ == "__main__":
    main()
