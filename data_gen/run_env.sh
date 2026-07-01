#!/usr/bin/env bash
# Wrapper that sets up the environment (chromium path + libasound) and runs a node script.
# Usage: ./run_env.sh generate.mjs --game coin_collection --episodes 3
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PW_CHROME="${PW_CHROME:-$HOME/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome}"
export LD_LIBRARY_PATH="$HERE/.condalibs/lib:${LD_LIBRARY_PATH:-}"
cd "$HERE"
exec node "$@"
