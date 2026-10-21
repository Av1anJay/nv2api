#!/bin/sh
# live smoke test against the real NVIDIA upstream with whatever key is set
set -u
BASE="${N2A_URL:-http://127.0.0.1:8211}"
echo "== models"
curl -s -m 25 "$BASE/v1/models" -o /tmp/nv2api_models.json -w "HTTP %{http_code}\n"
python3.12 - <<'PY'
import json
d = json.load(open('/tmp/nv2api_models.json'))
ids = [m['id'] for m in d.get('data', [])]
print('count', len(ids))
print('has cosmos3-nano-reasoner:', 'nvidia/cosmos3-nano-reasoner' in ids)
print('sample', ids[:4])
PY
echo "== stream chat"
curl -s -N -m 40 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"cosmos3-nano-reasoner","stream":true,"messages":[{"role":"user","content":"hi"}]}' \
  -o /tmp/nv2api_stream.txt -w "HTTP %{http_code}\n"
head -c 300 /tmp/nv2api_stream.txt
echo
echo "== server log tail"
tail -n 8 /tmp/nv2api.log
