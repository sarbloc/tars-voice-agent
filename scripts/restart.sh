#!/usr/bin/env bash
# Restart the tars services. The services are systemd --user units, so
# no sudo is needed. Use this after editing .env on the box without
# committing.
set -euo pipefail

for svc in tars-voice-agent.service tars-token-server.service; do
    echo "restarting $svc"
    systemctl --user restart "$svc"
done
