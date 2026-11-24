#!/bin/bash
# sdk.sh <toolchain.zip> <dir>: unpacks the vendor's IPU toolchain, hash-checked.
#
# The toolchain is SigmaStar's SGS_IPU_SDK vS1.1.6 ("S1" is the infinity6c's
# IPU). It is confidential and is not in this repository; the release runner
# has it, as the "IPU SDK-20250308T102046Z-001.zip" it was handed out as.
set -euo pipefail
sha=e8c98181df2136de4ed2d8c64cc7aaaed126606065b6d3f69524c11d34768b37
echo "$sha  $1" | sha256sum -c --quiet
[ -f "$2/SGS_IPU_SDK_vS1.1.6/cfg_env.sh" ] && exit 0
mkdir -p "$2"
unzip -p "$1" 'IPU SDK/Sigmastar_SDK_vS1.1.6.zip' > "$2/inner.zip"
unzip -p "$2/inner.zip" '*/SGS_IPU_SDK_vS1.1.6.tar.xz' | tar xJ -C "$2"
rm "$2/inner.zip"
