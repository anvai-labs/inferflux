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
#       [--rocm /opt/rocm]
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
ROCM_TAG="rocm-7.2.0" # The patch is a backport onto exactly this tag...
ROCM_COMMIT="fc0010cf6a5a972d42b276df946510f30343d493" # ...at this commit.
ROCDXG_TAG="v1.2.2" # Fixed until a second package checksum is pinned below.
ROCM_ROOT="/opt/rocm"

# Pinned prebuilt thunk packages (sha256 of the .deb from the GitHub release).
declare -A ROCDXG_SHA256=(
  [v1.2.2]="28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541"
)

usage() { # Help on stdout for --help, on stderr for usage errors.
  if [[ "${1:-0}" -eq 0 ]]; then sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  else sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//' >&2; fi
  exit "${1:-0}"
}
die() { echo "build_patched_rocr: $*" >&2; exit 1; }
need_value() {
  # Values may start with '-' (a path); only another option is refused.
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "$1 needs a value" >&2; usage 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) need_value "$@"; OUT="$2"; shift 2 ;;
    --work) need_value "$@"; WORK="$2"; shift 2 ;;
    --rocm) need_value "$@"; ROCM_ROOT="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
done

[[ -n "$OUT" ]] || usage 2
[[ -n "${ROCDXG_SHA256[$ROCDXG_TAG]:-}" ]] ||
  die "no pinned sha256 for librocdxg $ROCDXG_TAG; add one to ROCDXG_SHA256"
for tool in git curl cmake dpkg-deb readelf ldd nproc find sed grep cut paste ls mktemp realpath sha256sum; do
  command -v "$tool" >/dev/null || die "missing required tool: $tool"
done
# The backport and its ABI assumptions are for ROCm 7.2.0 exactly: refuse to
# drop a 7.2.0 runtime under a different host HIP.
# The backoff targets WSL2/dxg (no interrupt-backed events). On native Linux
# it would only add up to 200 us to polling-forced batches: refuse.
[[ -e /dev/dxg ]] || die "no /dev/dxg: this patched runtime is for WSL2 only"
HOST_ROCM="$(cat "${ROCM_ROOT}/.info/version" 2>/dev/null || true)"
[[ "$HOST_ROCM" =~ ^7\.2\.0([-+.~].*)?$ ]] ||
  die "host ROCm at ${ROCM_ROOT} is '${HOST_ROCM:-unknown}', not 7.2.0"
# The server loads HIP via /opt/rocm and the loader configuration, whatever
# --rocm says: that install must be 7.2.0 as well.
SYS_ROCM="$(cat /opt/rocm/.info/version 2>/dev/null || true)"
[[ "$SYS_ROCM" =~ ^7\.2\.0([-+.~].*)?$ ]] ||
  die "/opt/rocm (which the server loads HIP from) is '${SYS_ROCM:-missing}', not 7.2.0"
# Absolute path: the build runs in a clean environment (see below), where a
# cmake from ~/.local/bin or a venv would not be on PATH.
CMAKE="$(command -v cmake)"
CMAKE="$(cd "$(dirname "$CMAKE")" && pwd)/$(basename "$CMAKE")"
for tool in clang llvm-objcopy; do
  [[ -x "${ROCM_ROOT}/llvm/bin/${tool}" ]] ||
    die "missing ${ROCM_ROOT}/llvm/bin/${tool} (ROCm LLVM)"
done

# Markers: the script only ever deletes inside a work dir it created, and
# only replaces libraries in an output dir it staged. The deploy step copies
# libraries and PATCHED_ROCR.json but never these markers.
WORK_MARK=".build_patched_rocr-work"
OUT_MARK=".build_patched_rocr-out"
if [[ -z "$WORK" ]]; then
  WORK="$(mktemp -d -t patched-rocr.XXXXXX)"
  touch "${WORK}/${WORK_MARK}"
  # Auto-created: removed after success, or after a failure before any
  # source was fetched; otherwise kept (and named) for its logs/checkout.
  trap 'rc=$?; if [[ $rc -eq 0 || ! -d "${WORK}/rocm-systems" ]]; then rm -rf "$WORK";
        else echo "build_patched_rocr: work dir kept for its logs: ${WORK}" >&2; fi' EXIT
elif [[ -d "$WORK" && ! -f "${WORK}/${WORK_MARK}" ]] &&
     [[ -n "$(ls -A "$WORK")" ]]; then
  die "${WORK} is not empty and was not created by this script; pass an empty or new --work directory"
else
  mkdir -p "$WORK" && touch "${WORK}/${WORK_MARK}"
fi
if [[ -d "$OUT" && ! -f "${OUT}/${OUT_MARK}" && -n "$(ls -A "$OUT")" ]]; then
  die "${OUT} is not empty and was not staged by this script; use an empty or new --out directory (deploy with the documented backup step)"
fi
mkdir -p "$OUT"
WORK="$(cd "$WORK" && pwd)"
OUT="$(cd "$OUT" && pwd)"
SRC="${WORK}/rocm-systems"
BUILD="${WORK}/build"
echo "build_patched_rocr: work dir ${WORK}"

# 1. Source: rocr-runtime only, at the exact tag, with the backport applied.
# A partial or foreign checkout (e.g. an interrupted earlier run) is
# discarded rather than trusted.
if [[ -e "$SRC" ]] &&
   { [[ ! -d "${SRC}/.git" ]] ||
     [[ ! -f "${SRC}/projects/rocr-runtime/CMakeLists.txt" ]] ||
     [[ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" != "$ROCM_COMMIT" ]]; }; then
  rm -rf "$SRC"
fi
if [[ ! -d "${SRC}/.git" ]]; then
  # Fetched by the pinned commit, not the tag, so a moved tag cannot break
  # (or silently change) the build.
  git init --quiet "$SRC"
  git -C "$SRC" remote add origin https://github.com/ROCm/rocm-systems.git
  git -C "$SRC" sparse-checkout set projects/rocr-runtime
  git -C "$SRC" fetch --quiet --depth 1 --filter=blob:none origin "$ROCM_COMMIT"
  git -C "$SRC" -c advice.detachedHead=false checkout --quiet FETCH_HEAD
fi
git -C "$SRC" checkout --quiet -- .
git -C "$SRC" clean --quiet -fd -- projects/rocr-runtime
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$ROCM_COMMIT" ]] ||
  die "checkout is $(git -C "$SRC" rev-parse HEAD), expected ${ROCM_COMMIT} (${ROCM_TAG})"
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
LIBHSA_FOUND="$(find "$BUILD" -name 'libhsa-runtime64.so.1.*' -type f 2>/dev/null || true)"
[[ -n "$LIBHSA_FOUND" ]] || die "libhsa-runtime64 not produced"
# Exactly one, so what is staged and checksummed never depends on find order.
[[ "$LIBHSA_FOUND" != *$'\n'* ]] ||
  die "more than one libhsa-runtime64 produced: ${LIBHSA_FOUND//$'\n'/, }"
LIBHSA="$LIBHSA_FOUND"
# Like the stock runtime: no RUNPATH/RPATH (an empty element would make the
# loader search the service's working directory), and every dependency
# (libelf, libdrm, libdrm_amdgpu, libnuma, librocprofiler-register) must
# resolve through the system loader configuration.
# Captured first: under pipefail, grep -q closing the pipe early could make
# the check pass by mistake.
DYNAMIC="$(readelf -d "$LIBHSA")"
if grep -qE '\((RUNPATH|RPATH)\)' <<<"$DYNAMIC"; then
  die "built libhsa-runtime64 carries a RUNPATH/RPATH"
fi
# In a clean environment, as the service sees it: only the loader
# configuration, not this shell's LD_LIBRARY_PATH.
LDD_OUT="$("${clean_env[@]}" ldd "$LIBHSA")"
MISSING="$(grep 'not found' <<<"$LDD_OUT" || true)"
[[ -z "$MISSING" ]] || die "unresolved dependencies: ${MISSING}"
# The ROCm-provided dependency must come from a 7.2.0 install checked above:
# --rocm or /opt/rocm (what the clean loader configuration resolves).
PROFREG="$(sed -n 's/.*librocprofiler-register[^ ]* => \([^ ]*\).*/\1/p' <<<"$LDD_OUT")"
# (Optional in ROCr's CMake: only checked when it is linked.)
if [[ -n "$PROFREG" ]]; then
  PROFREG_REAL="$(realpath "$PROFREG")"
  if [[ "$PROFREG_REAL" != "$(realpath "$ROCM_ROOT")"/* &&
        "$PROFREG_REAL" != "$(realpath /opt/rocm)"/* ]]; then
    die "librocprofiler-register resolves to ${PROFREG}, outside ${ROCM_ROOT} and /opt/rocm"
  fi
fi
CXX_USED="$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "${BUILD}/CMakeCache.txt")"
CXX_VERSION="unknown"
if [[ -x "$CXX_USED" ]]; then # Informational: never fails the build.
  CXX_BANNER="$("$CXX_USED" --version 2>/dev/null || true)"
  CXX_VERSION="${CXX_BANNER%%$'\n'*}"
fi

# 3. Prebuilt WSL thunk: download, verify, extract (never installed).
DEB_DIR="${WORK}/rocdxg"
DEB="${DEB_DIR}/rocdxg-roct_${ROCDXG_TAG#v}_amd64.deb"
mkdir -p "$DEB_DIR"
if [[ ! -f "$DEB" ]]; then
  # Public release asset: no GitHub login needed; the checksum below is the
  # trust anchor.
  curl -fsSL --connect-timeout 30 --max-time 600 --retry 3 -o "${DEB}.part" \
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
LIBDXG="$(find "${DEB_DIR}/x" -name 'librocdxg.so.*.*' -type f 2>/dev/null | head -1 || true)"
[[ -n "$LIBDXG" ]] || die "librocdxg not found in $(basename "$DEB")"
# ROCr dlopens it as shipped. AMD's build carries RUNPATH /opt/rocm/lib; an
# absolute system path is fine, but an empty or relative element (which
# would search the service's working directory) is refused.
# RPATH and RUNPATH may both be present: check every entry of each.
DXG_RUNPATH="$(readelf -d "$LIBDXG" | sed -n 's/.*(\(RUNPATH\|RPATH\)).*\[\(.*\)\].*/\2/p' | paste -sd: -)"
if [[ -n "$DXG_RUNPATH" ]]; then
  IFS=':' read -r -a dxg_entries <<<"${DXG_RUNPATH}:"
  [[ "$DXG_RUNPATH" != *: && "$DXG_RUNPATH" != :* && "$DXG_RUNPATH" != *::* ]] ||
    die "librocdxg RUNPATH '${DXG_RUNPATH}' has an empty element"
  for entry in "${dxg_entries[@]}"; do
    [[ "$entry" == /* ]] || die "librocdxg RUNPATH '${DXG_RUNPATH}' has a relative element"
  done
fi
DXG_MISSING="$(grep 'not found' <<<"$("${clean_env[@]}" ldd "$LIBDXG")" || true)"
[[ -z "$DXG_MISSING" ]] || die "librocdxg has unresolved dependencies: ${DXG_MISSING}"

# 4. Stage with the sonames the loader looks up, assembled completely in
# ${WORK}/stage first. PATCHED_ROCR.json is the completion marker: it is
# removed from OUT before the libraries are replaced and renamed into place
# last, so an OUT without it is incomplete and must not be deployed.
STAGE="${WORK}/stage"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp "$LIBHSA" "${STAGE}/"
link_as() { # link_as TARGET NAME: skip when the file already has that name.
  [[ "$1" == "$2" ]] || ln -sfn "$1" "${STAGE}/$2"
}
link_as "$(basename "$LIBHSA")" libhsa-runtime64.so.1
cp "$LIBDXG" "${STAGE}/"
DXG_SONAME="$(readelf -d "$LIBDXG" | sed -n 's/.*(SONAME).*\[\(.*\)\].*/\1/p')"
[[ -n "$DXG_SONAME" ]] || die "librocdxg has no SONAME"
link_as "$(basename "$LIBDXG")" "$DXG_SONAME"
link_as "$(basename "$LIBDXG")" librocdxg.so # ROCr dlopens this name.

sha() { sha256sum "$1" | cut -d' ' -f1; }
json_str() { # Escape for a JSON string (backslash, quote, control chars).
  local s="${1//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf '%s' "$s" | tr '\000-\037\177' ' '
}
cat > "${STAGE}/PATCHED_ROCR.json" <<EOF
{
  "rocm_systems_tag": "${ROCM_TAG}",
  "host_rocm": "$(json_str "$HOST_ROCM")",
  "host_compiler": "$(json_str "$CXX_VERSION")",
  "rocm_systems_commit": "$(git -C "$SRC" rev-parse HEAD)",
  "upstream_fix": "ROCm/rocm-systems#7898 (46558b7af4dc79b8b8014619c1afdb82db079a9f)",
  "patch_sha256": "$(sha "$PATCH")",
  "rocdxg_tag": "${ROCDXG_TAG}",
  "rocdxg_deb_sha256": "${ROCDXG_SHA256[$ROCDXG_TAG]}",
  "libhsa_sha256": "$(sha "${STAGE}/$(basename "$LIBHSA")")",
  "librocdxg_sha256": "$(sha "${STAGE}/$(basename "$LIBDXG")")",
  "required_env": {"HSA_ENABLE_DXG_DETECTION": "1"}
}
EOF
# Swap the complete set into OUT, manifest last.
rm -f "${OUT}/PATCHED_ROCR.json" "${OUT}/.PATCHED_ROCR.json.tmp"
rm -f "${OUT}"/libhsa-runtime64.so* "${OUT}"/librocdxg.so*
touch "${OUT}/${OUT_MARK}"
cp -a "${STAGE}"/lib* "${OUT}/"
cp "${STAGE}/PATCHED_ROCR.json" "${OUT}/.PATCHED_ROCR.json.tmp"
mv "${OUT}/.PATCHED_ROCR.json.tmp" "${OUT}/PATCHED_ROCR.json"
rm -rf "$STAGE"
echo "build_patched_rocr: staged into ${OUT}"
ls -l "$OUT"
