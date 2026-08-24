#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=${1:-config/server.yaml}
BUILD_DIR=${BUILD_DIR:-build}

if [ ! -d "$BUILD_DIR" ]; then
  ./scripts/build.sh
fi

"$BUILD_DIR"/inferfluxd --config "$CONFIG_FILE"
