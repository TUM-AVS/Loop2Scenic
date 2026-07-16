#!/bin/bash
# NON-DESTRUCTIVE codeICL 145->245: decompose the 100 missing scenarios, write a SEPARATE 245 JSON,
# and build a SEPARATE Milvus collection `scenario_components_245`. The existing 922-from-145 file and
# `scenario_components` collection are left untouched (so the 245 version is cleanly deletable).
# Log: eval/results/rag/logs/codeicl_245.log
cd /home/yuan/aaai/Loop2Scenic || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate loop2scenic
set -a; source .env; set +a
export PYTHONPATH=.

echo "== [$(date '+%H:%M:%S')] STEP 1: decompose 100 missing scenarios -> 245 JSON =="
timeout 3600 python scripts/decompose_full.py || { echo "[ABORT] decompose failed"; exit 1; }

if [ ! -f data/raw_snippets/recovered_scenario_components_245.json ]; then
  echo "[ABORT] 245 JSON not written"; exit 1
fi

echo
echo "== [$(date '+%H:%M:%S')] STEP 2: build SEPARATE collection scenario_components_245 =="
timeout 1800 python scripts/insert_scenario_components.py \
  --collection scenario_components_245 \
  --json data/raw_snippets/recovered_scenario_components_245.json

echo
echo "== [$(date '+%H:%M:%S')] verify both collections coexist =="
python3 -c "
from pymilvus import connections, utility, Collection
from src.config import get_config
c=get_config(); connections.connect(uri=f'http://{c.vector_db.host}:{c.vector_db.port}')
for name in ['scenario_components','scenario_components_245']:
    if utility.has_collection(name):
        col=Collection(name); col.load(); print(f'  {name}: {col.num_entities} entities')
    else:
        print(f'  {name}: MISSING')
" 2>/dev/null
echo "CODEICL_245_DONE"
