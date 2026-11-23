#!/usr/bin/env bash
# Deploys the app to the Duke VM: copies the app, the scoring code and the saved models, builds
# the image there with rootless Podman, and restarts the user service that runs it on
# 127.0.0.1:8081. Caddy on the VM routes the public hostname to that port (jonasneves/infra, duke/).
#
#   TYPESAFE_API_KEY=... GEMINI_API_KEY=... ./deploy.sh        # each key enables its hosted model in the app
set -euo pipefail
cd "$(dirname "$0")"
HOST="${DUKE_HOST:-jd599@radchat.colab.duke.edu}"

rsync -az --delete --relative app.py gate.py decision.py requirements.txt Dockerfile connector-gate.container \
  models/baseline.joblib models/thresholds.json models/distilbert results/metrics.csv "$HOST:connector-gate/"
if [ -n "${TYPESAFE_API_KEY:-}${GEMINI_API_KEY:-}" ]; then
  printf 'TYPESAFE_API_KEY=%s\nGEMINI_API_KEY=%s\n' "${TYPESAFE_API_KEY:-}" "${GEMINI_API_KEY:-}" |
    ssh "$HOST" 'umask 077; cat > connector-gate/.env'
fi
ssh "$HOST" 'set -e
  touch connector-gate/.env && chmod 600 connector-gate/.env
  podman build -q -t connector-gate connector-gate
  mkdir -p ~/.config/containers/systemd
  cp connector-gate/connector-gate.container ~/.config/containers/systemd/
  systemctl --user daemon-reload
  systemctl --user restart connector-gate
  for i in $(seq 30); do curl -sf -o /dev/null http://127.0.0.1:8081/ && break; sleep 2; done
  curl -s -o /dev/null -w "app answers %{http_code} on the VM\n" http://127.0.0.1:8081/'
