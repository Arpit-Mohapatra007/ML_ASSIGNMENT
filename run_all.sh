set -eo pipefail
pip install -r requirements.txt
mkdir -p reports/logs
for s in audit split baselines train rohit_sim part2 shift_sim final_eval external ship ood_check ceo_math; do
    python -u $s.py 2>&1 | tee reports/logs/$s.txt
done
