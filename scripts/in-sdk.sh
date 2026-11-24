#!/bin/bash
# in-sdk.sh <out> <command...>: a command in the toolchain's container, with the
# SDK at /sdk, this repository at /src and <out> at /out. $IMAGES, when set, is
# mounted at /images.
src=$(cd "$(dirname "$0")/.." && pwd)
out=$1; shift
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$src:/src:ro" \
	-v "$src/.cache/sdk/SGS_IPU_SDK_vS1.1.6:/sdk:ro" -v "$out:/out" \
	${IMAGES:+-v "$IMAGES:/images:ro"} sgsnpu-s1 \
	bash -c 'cd /sdk && . ./cfg_env.sh && cd /tmp && exec "$@"' _ "$@"
