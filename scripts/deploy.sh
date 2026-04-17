#!/usr/bin/env bash
# Pull latest main and restart the tars services if anything changed.
# Safe to run on a timer — exits fast when there's nothing new.
#
# The tars services are systemd --user units, so no sudo is needed.
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

for svc in tars-voice-agent.service tars-token-server.service; do
    log "restarting $svc"
    systemctl --user restart "$svc"
done

log "deploy complete"
