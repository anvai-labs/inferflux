#!/usr/bin/env bash
# Build a ROCr runtime (libhsa-runtime64) with the AsyncEventsLoop poll
# backoff (ROCm/rocm-systems#7898) backported, plus AMD's prebuilt WSL thunk
# (librocdxg), and stage both for a release's runtime/bin directory.
#
# Why: on WSL2 the stock ROCm 7.2.x runtime keeps two AsyncEventsLoop
# threads spinning at 100% CPU each for the life of every GPU process
# (ROCm/librocdxg#60). The upstream fix is merged but not in a released ROCm.
# See docs/ROCM_INSTALLATION_GUIDE_WSL.md ("Patched ROCr runtime").
#
# Usage:
#   scripts/rocm/build_patched_rocr.sh --out DIR [--work DIR]
#       [--rocm-tag rocm-7.2.0] [--rocdxg v1.2.2] [--rocm /opt/rocm]
#
# Nothing is installed system-wide. DIR receives libhsa-runtime64.so*,
# librocdxg.so* and PATCHED_ROCR.json. Deploy by copying them into the
# release's runtime/bin (already on the service's LD_LIBRARY_PATH) and setting
# HSA_ENABLE_DXG_DETECTION=1 for the service.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${SCRIPT_DIR}/patches/rocr-7.2.0-async-events-poll-backoff.patch"

OUT=""
WORK=""
ROCM_TAG="rocm-7.2.0"
ROCDXG_TAG="v1.2.2"
ROCM_ROOT="/opt/rocm"

# Pinned prebuilt thunk packages (sha256 of the .deb from the GitHub release).
declare -A ROCDXG_SHA256=(
  [v1.2.2]="28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541"
)

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "build_patched_rocr: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --rocm-tag) ROCM_TAG="$2"; shift 2 ;;
    --rocdxg) ROCDXG_TAG="$2"; shift 2 ;;
    --rocm) ROCM_ROOT="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
done

[[ -n "$OUT" ]] || usage 2
[[ "$ROCM_TAG" == "rocm-7.2.0" ]] ||
  die "the patch is a backport onto rocm-7.2.0; rebase it before using $ROCM_TAG"
[[ -n "${ROCDXG_SHA256[$ROCDXG_TAG]:-}" ]] ||
  die "no pinned sha256 for librocdxg $ROCDXG_TAG; add one to ROCDXG_SHA256"
for tool in git gh cmake dpkg-deb sha256sum; do
  command -v "$tool" >/dev/null || die "missing required tool: $tool"
done
for tool in clang llvm-objcopy; do
  [[ -x "${ROCM_ROOT}/llvm/bin/${tool}" ]] ||
    die "missing ${ROCM_ROOT}/llvm/bin/${tool} (ROCm LLVM)"
done

WORK="${WORK:-$(mktemp -d -t patched-rocr.XXXXXX)}"
mkdir -p "$WORK" "$OUT"
WORK="$(cd "$WORK" && pwd)"
OUT="$(cd "$OUT" && pwd)"
SRC="${WORK}/rocm-systems"
BUILD="${WORK}/build"
echo "build_patched_rocr: work dir ${WORK}"

# 1. Source: rocr-runtime only, at the exact tag, with the backport applied.
if [[ ! -d "${SRC}/.git" ]]; then
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$ROCM_TAG" --filter=blob:none \
    --sparse https://github.com/ROCm/rocm-systems.git "$SRC"
  git -C "$SRC" sparse-checkout set projects/rocr-runtime
fi
git -C "$SRC" checkout --quiet -- .
git -C "$SRC" clean --quiet -fd -- projects/rocr-runtime
git -C "$SRC" apply --check "$PATCH" || die "patch does not apply to $ROCM_TAG"
git -C "$SRC" apply "$PATCH"

# 2. Build libhsa-runtime64 only. ROCm's bundled LLVM ships no CMake package,
# and ROCr only needs two imported executables from it, so provide shims.
SHIM="${WORK}/llvm-shim"
mkdir -p "${SHIM}/clang" "${SHIM}/llvm"
cat > "${SHIM}/clang/ClangConfig.cmake" <<EOF
if(NOT TARGET clang)
  add_executable(clang IMPORTED GLOBAL)
  set_target_properties(clang PROPERTIES IMPORTED_LOCATION "${ROCM_ROOT}/llvm/bin/clang")
endif()
EOF
cat > "${SHIM}/llvm/LLVMConfig.cmake" <<EOF
if(NOT TARGET llvm-objcopy)
  add_executable(llvm-objcopy IMPORTED GLOBAL)
  set_target_properties(llvm-objcopy PROPERTIES IMPORTED_LOCATION "${ROCM_ROOT}/llvm/bin/llvm-objcopy")
endif()
EOF
# A clean environment keeps conda/venv compilers, sysroots and LDFLAGS out.
clean_env=(env -i "HOME=${HOME}" "PATH=/usr/local/bin:/usr/bin:/bin")
"${clean_env[@]}" cmake -S "${SRC}/projects/rocr-runtime" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_PREFIX_PATH="$ROCM_ROOT" \
  -DClang_DIR="${SHIM}/clang" -DLLVM_DIR="${SHIM}/llvm" \
  -DBUILD_ROCR_TESTS=OFF >"${WORK}/configure.log" 2>&1 ||
  die "configure failed; see ${WORK}/configure.log"
"${clean_env[@]}" cmake --build "$BUILD" -j"$(nproc)" >"${WORK}/build.log" 2>&1 ||
  die "build failed; see ${WORK}/build.log"
LIBHSA="$(find "${BUILD}/rocr/lib" -name 'libhsa-runtime64.so.1.*' -type f | head -1)"
[[ -n "$LIBHSA" ]] || die "libhsa-runtime64 not produced"

# 3. Prebuilt WSL thunk: download, verify, extract (never installed).
DEB_DIR="${WORK}/rocdxg"
DEB="${DEB_DIR}/rocdxg-roct_${ROCDXG_TAG#v}_amd64.deb"
mkdir -p "$DEB_DIR"
if [[ ! -f "$DEB" ]]; then
  gh release download "$ROCDXG_TAG" -R ROCm/librocdxg \
    -p "$(basename "$DEB")" -D "$DEB_DIR" --clobber
fi
echo "${ROCDXG_SHA256[$ROCDXG_TAG]}  ${DEB}" | sha256sum --check --quiet ||
  die "librocdxg package checksum mismatch"
rm -rf "${DEB_DIR}/x"
dpkg-deb -x "$DEB" "${DEB_DIR}/x"
LIBDXG="$(find "${DEB_DIR}/x" -name 'librocdxg.so.*.*' -type f | head -1)"
[[ -n "$LIBDXG" ]] || die "librocdxg not found in $(basename "$DEB")"

# 4. Stage with the sonames the loader looks up.
rm -f "${OUT}"/libhsa-runtime64.so* "${OUT}"/librocdxg.so*
cp "$LIBHSA" "${OUT}/"
ln -s "$(basename "$LIBHSA")" "${OUT}/libhsa-runtime64.so.1"
cp "$LIBDXG" "${OUT}/"
ln -s "$(basename "$LIBDXG")" "${OUT}/librocdxg.so.1"
ln -s "$(basename "$LIBDXG")" "${OUT}/librocdxg.so" # ROCr dlopens this name.

sha() { sha256sum "$1" | cut -d' ' -f1; }
cat > "${OUT}/PATCHED_ROCR.json" <<EOF
{
  "rocm_systems_tag": "${ROCM_TAG}",
  "rocm_systems_commit": "$(git -C "$SRC" rev-parse HEAD)",
  "upstream_fix": "ROCm/rocm-systems#7898 (46558b7af4dc79b8b8014619c1afdb82db079a9f)",
  "patch_sha256": "$(sha "$PATCH")",
  "rocdxg_tag": "${ROCDXG_TAG}",
  "rocdxg_deb_sha256": "${ROCDXG_SHA256[$ROCDXG_TAG]}",
  "libhsa_sha256": "$(sha "${OUT}/$(basename "$LIBHSA")")",
  "librocdxg_sha256": "$(sha "${OUT}/$(basename "$LIBDXG")")",
  "required_env": {"HSA_ENABLE_DXG_DETECTION": "1"}
}
EOF
echo "build_patched_rocr: staged into ${OUT}"
ls -l "$OUT"
