#!/bin/bash
# race_check.sh -- data-race check of cpu_only, cpu_sparse and cpu_sparse_v2.
#
#   make race-check        (builds *_clangtsan with clang + libomp, then runs this)
#
# The programs are built with clang's ThreadSanitizer and LLVM's OpenMP
# runtime. Archer (libarcher.so, shipped with libomp) reports OpenMP's
# barriers and reductions to ThreadSanitizer; without it every OpenMP program
# shows false races. GCC's libgomp has no such interface, so this check
# needs clang (Ubuntu: apt install clang-18 libomp-18-dev).
#
# Exit status 1 if any run reports a data race or fails.
#
# Generated with assistance of claude code

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_DIR="${LLVM_DIR:-/usr/lib/llvm-18}"
ARCHER="${ARCHER:-$LLVM_DIR/lib/libarcher.so}"
[ -f "$ARCHER" ] || { echo "libarcher.so not found at $ARCHER (install libomp-dev, or set ARCHER=)"; exit 2; }
for p in cpu_only cpu_sparse cpu_sparse_v2; do
    [ -x "$ROOT/${p}_clangtsan" ] || { echo "missing ${p}_clangtsan: run make race-check"; exit 2; }
done
[ -x "$ROOT/tools/gen_fem" ] || { echo "missing tools/gen_fem: run make"; exit 2; }

export OMP_TOOL_LIBRARIES="$ARCHER"
export ARCHER_OPTIONS="verbose=1"
export TSAN_OPTIONS="ignore_noninstrumented_modules=1 halt_on_error=0 exitcode=66"
export LD_LIBRARY_PATH="$LLVM_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SAVE_SOLUTION=0
unset OMP_THREAD_LIMIT OMP_DYNAMIC REORDER PRECOND FIRST_TOUCH BALANCE THREAD_GRAIN
# ThreadSanitizer needs the classic address-space layout on recent kernels
NORAND=""
command -v setarch > /dev/null && NORAND="setarch $(uname -m) -R"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK" || exit 2
"$ROOT/tools/gen_fem" poisson2d 20 -o p2d.mtx > /dev/null
"$ROOT/tools/gen_fem" elasticity 3 --shuffle 2 -o el3.mtx > /dev/null
cp "$ROOT/qa/data/494_bus.mtx" .

PASSED=0
FAILED=0
ARCHER_SEEN=0
check() {   # label, then [VAR=value ...] program args
    local label="$1"; shift
    local out rc
    out=$(env "$@" 2>&1); rc=$?
    grep -qi "archer detected" <<< "$out" && ARCHER_SEEN=1
    if grep -q "ThreadSanitizer: data race" <<< "$out"; then
        printf "FAIL  %s: data race\n" "$label"
        grep -m1 -A12 "WARNING: ThreadSanitizer" <<< "$out" | sed 's/^/      /'
        FAILED=$((FAILED + 1))
    elif [ $rc -ne 0 ] || ! grep -q "Relative Residual" <<< "$out"; then
        printf "FAIL  %s: exit %d\n" "$label" "$rc"
        tail -5 <<< "$out" | sed 's/^/      /'
        FAILED=$((FAILED + 1))
    else
        printf "PASS  %s\n" "$label"
        PASSED=$((PASSED + 1))
    fi
}

for t in 2 4; do
    for m in p2d el3 494_bus; do
        check "cpu_only $m, $t threads" OMP_NUM_THREADS=$t $NORAND "$ROOT/cpu_only_clangtsan" "$m.mtx"
        for pc in none jacobi; do
            check "cpu_sparse $m, $t threads, PRECOND=$pc" OMP_NUM_THREADS=$t PRECOND=$pc \
                $NORAND "$ROOT/cpu_sparse_clangtsan" "$m.mtx"
            for ft in 0 1; do
                check "cpu_sparse_v2 $m, $t threads, PRECOND=$pc FIRST_TOUCH=$ft" OMP_NUM_THREADS=$t \
                    PRECOND=$pc FIRST_TOUCH=$ft $NORAND "$ROOT/cpu_sparse_v2_clangtsan" "$m.mtx"
            done
        done
    done
done
check "cpu_sparse_v2 p2d, OMP_THREAD_LIMIT=2 below 4 threads" OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=2 \
    $NORAND "$ROOT/cpu_sparse_v2_clangtsan" p2d.mtx
check "cpu_sparse_v2 p2d, BALANCE=rows THREAD_GRAIN=64" OMP_NUM_THREADS=4 BALANCE=rows THREAD_GRAIN=64 \
    $NORAND "$ROOT/cpu_sparse_v2_clangtsan" p2d.mtx

if [ $ARCHER_SEEN = 0 ]; then
    echo "FAIL  Archer was not loaded, so OpenMP synchronization was invisible to ThreadSanitizer"
    FAILED=$((FAILED + 1))
fi
printf "\n%d passed, %d failed (ThreadSanitizer + Archer)\n" "$PASSED" "$FAILED"
[ "$FAILED" -eq 0 ]
