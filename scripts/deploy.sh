#!/usr/bin/env bash
# Pull latest main and restart the tars services if anything changed.
# Safe to run on a timer — exits fast when there's nothing new.
#
# Requires /etc/sudoers.d/tars-deploy to allow passwordless
# `systemctl restart tars-*.service`. See scripts/install-sudoers.sh.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

log() { printf '[deploy %s] %s\n' "$(date -Iseconds)" "$*"; }

current="$(git rev-parse HEAD)"
git fetch --quiet origin main
target="$(git rev-parse origin/main)"

if [[ "$current" == "$target" ]]; then
    log "already at $target, nothing to do"
    exit 0
fi

log "updating $current -> $target"
git pull --ff-only --quiet origin main

# Restart services in dependency order.
for svc in tars-voice-agent.service tars-token-server.service; do
    log "restarting $svc"
    sudo -n /usr/bin/systemctl restart "$svc"
done

log "deploy complete"
