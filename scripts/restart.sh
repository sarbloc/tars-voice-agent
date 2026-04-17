#!/usr/bin/env bash
# Restart the tars services without pulling. Useful after editing .env
# or manually changing files on the box.
set -euo pipefail

for svc in tars-voice-agent.service tars-token-server.service; do
    echo "restarting $svc"
    sudo -n /usr/bin/systemctl restart "$svc"
done
