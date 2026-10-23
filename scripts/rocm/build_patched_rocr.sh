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
#       [--rocdxg v1.2.2] [--rocm /opt/rocm]
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
ROCM_TAG="rocm-7.2.0" # The patch is a backport onto exactly this tag.
ROCDXG_TAG="v1.2.2"
ROCM_ROOT="/opt/rocm"

# Pinned prebuilt thunk packages (sha256 of the .deb from the GitHub release).
declare -A ROCDXG_SHA256=(
  [v1.2.2]="28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541"
)

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "build_patched_rocr: $*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || { echo "$1 needs a value" >&2; usage 2; }; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) need_value "$@"; OUT="$2"; shift 2 ;;
    --work) need_value "$@"; WORK="$2"; shift 2 ;;
    --rocdxg) need_value "$@"; ROCDXG_TAG="$2"; shift 2 ;;
    --rocm) need_value "$@"; ROCM_ROOT="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
done

[[ -n "$OUT" ]] || usage 2
[[ -n "${ROCDXG_SHA256[$ROCDXG_TAG]:-}" ]] ||
  die "no pinned sha256 for librocdxg $ROCDXG_TAG; add one to ROCDXG_SHA256"
for tool in git curl cmake dpkg-deb readelf sha256sum; do
  command -v "$tool" >/dev/null || die "missing required tool: $tool"
done
# The backport and its ABI assumptions are for ROCm 7.2.0 exactly: refuse to
# drop a 7.2.0 runtime under a different host HIP.
HOST_ROCM="$(cat "${ROCM_ROOT}/.info/version" 2>/dev/null || true)"
[[ "$HOST_ROCM" == "7.2.0" ]] ||
  die "host ROCm at ${ROCM_ROOT} is '${HOST_ROCM:-unknown}', not 7.2.0"
# Absolute path: the build runs in a clean environment (see below), where a
# cmake from ~/.local/bin or a venv would not be on PATH.
CMAKE="$(command -v cmake)"
CMAKE="$(cd "$(dirname "$CMAKE")" && pwd)/$(basename "$CMAKE")"
for tool in clang llvm-objcopy; do
  [[ -x "${ROCM_ROOT}/llvm/bin/${tool}" ]] ||
    die "missing ${ROCM_ROOT}/llvm/bin/${tool} (ROCm LLVM)"
done

if [[ -z "$WORK" ]]; then
  WORK="$(mktemp -d -t patched-rocr.XXXXXX)"
  trap 'rm -rf "$WORK"' EXIT # Auto-created: never leave it behind.
fi
mkdir -p "$WORK" "$OUT"
WORK="$(cd "$WORK" && pwd)"
OUT="$(cd "$OUT" && pwd)"
SRC="${WORK}/rocm-systems"
BUILD="${WORK}/build"
echo "build_patched_rocr: work dir ${WORK}"

# 1. Source: rocr-runtime only, at the exact tag, with the backport applied.
# A partial or foreign checkout (e.g. an interrupted earlier run) is
# discarded rather than trusted.
if [[ -d "${SRC}/.git" ]] &&
   { [[ ! -f "${SRC}/projects/rocr-runtime/CMakeLists.txt" ]] ||
     [[ "$(git -C "$SRC" describe --tags --exact-match 2>/dev/null)" != "$ROCM_TAG" ]]; }; then
  rm -rf "$SRC"
fi
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
rm -rf "$BUILD" # Never stage a library left over from an earlier build.
"${clean_env[@]}" "$CMAKE" -S "${SRC}/projects/rocr-runtime" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_PREFIX_PATH="$ROCM_ROOT" \
  -DClang_DIR="${SHIM}/clang" -DLLVM_DIR="${SHIM}/llvm" \
  -DCMAKE_SKIP_RPATH=ON \
  -DBUILD_ROCR_TESTS=OFF >"${WORK}/configure.log" 2>&1 ||
  die "configure failed; see ${WORK}/configure.log"
"${clean_env[@]}" "$CMAKE" --build "$BUILD" -j"$(nproc)" >"${WORK}/build.log" 2>&1 ||
  die "build failed; see ${WORK}/build.log"
LIBHSA="$(find "${BUILD}/rocr/lib" -name 'libhsa-runtime64.so.1.*' -type f | head -1)"
[[ -n "$LIBHSA" ]] || die "libhsa-runtime64 not produced"
# Like the stock runtime: no RUNPATH/RPATH (an empty element would make the
# loader search the service's working directory), and every dependency
# (libelf, libdrm, libdrm_amdgpu, libnuma, librocprofiler-register) must
# resolve through the system loader configuration.
if readelf -d "$LIBHSA" | grep -qE '\((RUNPATH|RPATH)\)'; then
  die "built libhsa-runtime64 carries a RUNPATH/RPATH"
fi
MISSING="$(ldd "$LIBHSA" | grep 'not found' || true)"
[[ -z "$MISSING" ]] || die "unresolved dependencies: ${MISSING}"
CXX_USED="$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "${BUILD}/CMakeCache.txt")"
CXX_VERSION="$("$CXX_USED" --version | head -1)"

# 3. Prebuilt WSL thunk: download, verify, extract (never installed).
DEB_DIR="${WORK}/rocdxg"
DEB="${DEB_DIR}/rocdxg-roct_${ROCDXG_TAG#v}_amd64.deb"
mkdir -p "$DEB_DIR"
if [[ ! -f "$DEB" ]]; then
  # Public release asset: no GitHub login needed; the checksum below is the
  # trust anchor.
  curl -fsSL -o "${DEB}.part" \
    "https://github.com/ROCm/librocdxg/releases/download/${ROCDXG_TAG}/$(basename "$DEB")" ||
    die "could not download $(basename "$DEB")"
  mv "${DEB}.part" "$DEB"
fi
if ! echo "${ROCDXG_SHA256[$ROCDXG_TAG]}  ${DEB}" | sha256sum --check --quiet; then
  rm -f "$DEB" # Never keep a bad cached download: the next run refetches.
  die "librocdxg package checksum mismatch"
fi
rm -rf "${DEB_DIR}/x"
dpkg-deb -x "$DEB" "${DEB_DIR}/x"
LIBDXG="$(find "${DEB_DIR}/x" -name 'librocdxg.so.*.*' -type f | head -1)"
[[ -n "$LIBDXG" ]] || die "librocdxg not found in $(basename "$DEB")"

# 4. Stage with the sonames the loader looks up.
rm -f "${OUT}"/libhsa-runtime64.so* "${OUT}"/librocdxg.so* "${OUT}/PATCHED_ROCR.json"
cp "$LIBHSA" "${OUT}/"
ln -s "$(basename "$LIBHSA")" "${OUT}/libhsa-runtime64.so.1"
cp "$LIBDXG" "${OUT}/"
DXG_SONAME="$(readelf -d "$LIBDXG" | sed -n 's/.*(SONAME).*\[\(.*\)\].*/\1/p')"
[[ -n "$DXG_SONAME" ]] || die "librocdxg has no SONAME"
ln -s "$(basename "$LIBDXG")" "${OUT}/${DXG_SONAME}"
ln -s "$(basename "$LIBDXG")" "${OUT}/librocdxg.so" # ROCr dlopens this name.

sha() { sha256sum "$1" | cut -d' ' -f1; }
cat > "${OUT}/PATCHED_ROCR.json" <<EOF
{
  "rocm_systems_tag": "${ROCM_TAG}",
  "host_rocm": "${HOST_ROCM}",
  "host_compiler": "${CXX_VERSION}",
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
