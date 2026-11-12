#!/bin/bash
# run_tests.sh -- correctness tests for cpu_only, cpu_sparse, cpu_sparse_v2 and tools/.
#
#   make test            (builds first)
#   bash qa/run_tests.sh
#
# Prints PASS/FAIL per check and exits 1 if any check fails. Runs in a
# temporary folder; needs only the binaries and qa/data/.
#
# Generated with assistance of claude code

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/qa/data"
GEN="$ROOT/tools/gen_fem"
CHECK="$ROOT/tools/check_solution"
STUDY="$ROOT/tools/rcm_study"
DENSE="$ROOT/cpu_only"
V1="$ROOT/cpu_sparse"
V2="$ROOT/cpu_sparse_v2"
for b in "$GEN" "$CHECK" "$STUDY" "$DENSE" "$V1" "$V2"; do
    [ -x "$b" ] || { echo "missing $b: run make first"; exit 2; }
done

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK" || exit 2
unset OMP_THREAD_LIMIT OMP_DYNAMIC REORDER PRECOND MAX_ITERS TOL SAVE_SOLUTION \
      FIRST_TOUCH BALANCE THREAD_GRAIN OMP_PROC_BIND OMP_PLACES
export OMP_NUM_THREADS=4

PASSED=0
FAILED=0
pass() { printf "PASS  %s\n" "$1"; PASSED=$((PASSED + 1)); }
fail() { printf "FAIL  %s\n" "$1"; [ $# -gt 1 ] && printf "      %s\n" "$2"; FAILED=$((FAILED + 1)); }
section() { printf "\n== %s\n" "$1"; }

iters()   { awk '/^Iterations Taken:/ {print $3}' <<< "$1"; }
resid()   { awk '/^Relative Residual:/ {print $3}' <<< "$1"; }
nthreads(){ awk '/^Threads:/ {print $2}' <<< "$1"; }
bw()      { awk '/^Bandwidth:/ {print $2, $4}' <<< "$1"; }
lt()      { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a != "" && a + 0 < b + 0) }'; }
near()    { awk -v a="$1" -v b="$2" -v d="$3" 'BEGIN { x = a - b; if (x < 0) x = -x; exit !(a != "" && x <= d) }'; }
sol() {   # program matrix-stem -> solution file it writes
    case "$(basename "$1")" in
        cpu_only)      echo "$2_dense_solution.txt" ;;
        cpu_sparse)    echo "$2_csr_solution.txt" ;;
        cpu_sparse_v2) echo "$2_csr_v2_solution.txt" ;;
    esac
}
run() { OUT=$(env "$@" 2>&1); RC=$?; }   # run [VAR=value ...] program args
check() { CHK=$("$CHECK" "$@" 2>&1); CRC=$?; }   # check_solution args; CRC = its exit status

# ---------------------------------------------------------------- inputs ----
"$GEN" poisson1d 100 -o p1d.mtx > /dev/null
"$GEN" poisson2d 30 -o p2d.mtx > /dev/null
"$GEN" poisson2d 30 --shuffle 7 -o p2d_shuf.mtx > /dev/null
"$GEN" poisson3d 8 -o p3d.mtx > /dev/null
"$GEN" elasticity 4 -o el4.mtx > /dev/null
"$GEN" elasticity 4 --shuffle 3 -o el4_shuf.mtx > /dev/null
cp "$DATA"/bcsstk01.mtx "$DATA"/nos4.mtx "$DATA"/494_bus.mtx "$DATA"/young1c.mtx.gz .
cp "$ROOT/Matrices/west0167.mtx" . 2> /dev/null
gzip -c p2d.mtx > p2dgz.mtx.gz
# exact answers for b = 1
awk 'BEGIN { n = 100; for (i = 0; i < n; i++) printf "%d %.17g\n", i, (i + 1) * (n - i) / 2 }' > p1d_exact.txt
{ echo "%%MatrixMarket matrix coordinate real symmetric"; echo "50 50 50"
  for i in $(seq 1 50); do echo "$i $i $(( (i - 1) % 10 + 1 ))"; done; } > diag.mtx
awk 'BEGIN { for (i = 0; i < 50; i++) printf "%d %.17g\n", i, 1 / (i % 10 + 1) }' > diag_exact.txt
printf '%%%%MatrixMarket matrix coordinate real symmetric\n1 1 1\n1 1 4\n' > one.mtx
echo "0 0.25" > one_exact.txt
# 1D Poisson n=5 with every diagonal 2 written as two entries 1 + 1
{ echo "%%MatrixMarket matrix coordinate real symmetric"; echo "5 5 14"
  for i in 1 2 3 4 5; do echo "$i $i 1"; echo "$i $i 1"; done
  for i in 2 3 4 5; do echo "$i $((i - 1)) -1"; done; } > dup.mtx
printf '0 2.5\n1 4\n2 4.5\n3 4\n4 2.5\n' > dup_exact.txt
printf '%%%%MatrixMarket matrix coordinate pattern symmetric\n4 4 4\n1 1\n2 2\n3 3\n4 4\n' > eye.mtx
printf '0 1\n1 1\n2 1\n3 1\n' > eye_exact.txt
printf '%%%%MatrixMarket matrix coordinate real general\n3 4 2\n1 1 1\n2 2 1\n' > rect.mtx

# ------------------------------------------------------------------ tests ---
section "generator"
run "$GEN" elasticity 3 --check -o el3.mtx
if [ $RC -eq 0 ] && grep -q "element check:.*OK" <<< "$OUT"; then pass "elasticity element: symmetric, rigid motions give zero force"
else fail "elasticity element check" "$OUT"; fi

section "known answers (all three solvers)"
for p in "$DENSE" "$V1" "$V2"; do
    name=$(basename "$p")
    for t in "p1d|TOL=1e-12 MAX_ITERS=2000|1e-8|1D Poisson, x_i = (i+1)(n-i)/2" \
             "diag|TOL=1e-12|1e-10|diagonal, x_i = 1/d_i" \
             "one|TOL=1e-12|1e-12|1x1 matrix" \
             "dup|TOL=1e-12|1e-10|repeated entries are summed" \
             "eye|TOL=1e-12|1e-12|pattern file (values 1)"; do
        IFS='|' read -r m envs rtol label <<< "$t"
        run $envs "$p" "$m.mtx"
        check "$m.mtx" "$(sol "$p" "$m")" --ref "${m}_exact.txt" --tol 1e-9 --ref-tol "$rtol"
        if [ $RC -eq 0 ] && [ $CRC -eq 0 ]; then pass "$name: $label"
        else fail "$name: $label" "$(tail -1 <<< "$CHK")"; fi
    done
done

section "the three solvers agree (plain CG, same matrix)"
for m in p2d el4; do
    run "$DENSE" "$m.mtx"; it0=$(iters "$OUT")
    for p in "$V1" "$V2"; do
        run "$p" "$m.mtx"; it=$(iters "$OUT")
        check "$m.mtx" "$(sol "$p" "$m")" --ref "$(sol "$DENSE" "$m")" --ref-tol 1e-8
        if [ "$it" = "$it0" ] && [ $CRC -eq 0 ]; then
            pass "$m: $(basename "$p") matches cpu_only ($it iterations, residual rechecked from file)"
        else fail "$m: $(basename "$p") vs cpu_only" "iters $it vs $it0; $(tail -1 <<< "$CHK")"; fi
    done
done

section "thread counts"
for p in "$V1" "$V2"; do
    ok=1; msg=""
    for t in 1 2 4 8; do
        run OMP_NUM_THREADS=$t "$p" p2d.mtx
        if [ "$(iters "$OUT")" != 43 ] || ! lt "$(resid "$OUT")" 1e-5 || [ "$(nthreads "$OUT")" != "$t" ]; then
            ok=0; msg="threads=$t: iters=$(iters "$OUT") resid=$(resid "$OUT") printed=$(nthreads "$OUT")"
        fi
    done
    [ $ok = 1 ] && pass "$(basename "$p"): same iterations and residual at 1, 2, 4, 8 threads" || fail "$(basename "$p") thread counts" "$msg"
done
run PRECOND=jacobi "$V2" 494_bus.mtx; cp "$(sol "$V2" 494_bus)" first.txt
run PRECOND=jacobi "$V2" 494_bus.mtx
cmp -s first.txt "$(sol "$V2" 494_bus)" && pass "cpu_sparse_v2: bit-identical answer when rerun at the same thread count" \
                                         || fail "cpu_sparse_v2 rerun differs"
for p in "$DENSE" "$V1" "$V2"; do
    for ft in 1 0; do
        [ "$(basename "$p")" != cpu_sparse_v2 ] && [ $ft = 0 ] && continue
        run OMP_NUM_THREADS=8 OMP_THREAD_LIMIT=2 FIRST_TOUCH=$ft "$p" p2d.mtx
        label="$(basename "$p")$([ "$(basename "$p")" = cpu_sparse_v2 ] && echo " FIRST_TOUCH=$ft"): OMP_THREAD_LIMIT=2 below OMP_NUM_THREADS=8"
        if [ $RC -eq 0 ] && lt "$(resid "$OUT")" 1e-5 && [ "$(nthreads "$OUT")" = 2 ]; then pass "$label"
        else fail "$label" "rc=$RC resid=$(resid "$OUT") threads=$(nthreads "$OUT")"; fi
    done
done
base=""; ok=1; msg=""
for ft in 0 1; do for bal in rows nnz; do for grain in 0 256; do
    run PRECOND=jacobi FIRST_TOUCH=$ft BALANCE=$bal THREAD_GRAIN=$grain "$V2" el4.mtx
    it=$(iters "$OUT"); [ -z "$base" ] && base=$it
    if [ $RC -ne 0 ] || ! lt "$(resid "$OUT")" 1e-5 || ! near "$it" "$base" 2; then
        ok=0; msg="FIRST_TOUCH=$ft BALANCE=$bal THREAD_GRAIN=$grain: rc=$RC iters=$it resid=$(resid "$OUT")"
    fi
done; done; done
[ $ok = 1 ] && pass "cpu_sparse_v2: every FIRST_TOUCH / BALANCE / THREAD_GRAIN combination converges" \
            || fail "cpu_sparse_v2 option combinations" "$msg"

section "reordering"
for p in "$V1" "$V2"; do
    for m in p2d_shuf el4_shuf; do
        run REORDER=none PRECOND=jacobi "$p" "$m.mtx"; it0=$(iters "$OUT"); cp "$(sol "$p" "$m")" none.txt
        run REORDER=rcm PRECOND=jacobi "$p" "$m.mtx"; it=$(iters "$OUT"); read -r before after <<< "$(bw "$OUT")"
        check "$m.mtx" "$(sol "$p" "$m")" --ref none.txt --ref-tol 1e-6
        if [ $CRC -eq 0 ] && near "$it" "$it0" 1 && [ "$after" -lt "$before" ]; then
            pass "$(basename "$p") $m: RCM keeps the answer and iterations, bandwidth $before -> $after"
        else fail "$(basename "$p") $m: RCM" "iters $it vs $it0, bandwidth $before -> $after; $(tail -1 <<< "$CHK")"; fi
    done
    run REORDER=auto "$p" p2d.mtx; a=$OUT
    run REORDER=auto "$p" p2d_shuf.mtx
    if grep -q "RCM skipped" <<< "$a" && grep -q "(after RCM)" <<< "$OUT"; then
        pass "$(basename "$p"): REORDER=auto skips RCM on an ordered grid and applies it to a shuffled one"
    else fail "$(basename "$p"): REORDER=auto" "$(grep Bandwidth <<< "$a") / $(grep Bandwidth <<< "$OUT")"; fi
done
run "$STUDY" --reps 1 --csv study.csv bcsstk01.mtx nos4.mtx 494_bus.mtx p2d_shuf.mtx el4_shuf.mtx
if [ $RC -eq 0 ] && ! grep -q MISMATCH study.csv; then pass "rcm_study: every ordering is a permutation and gives the same A*x"
else fail "rcm_study self-checks" "$OUT"; fi
ok=1; msg=""
for m in bcsstk01 nos4 494_bus p2d_shuf el4_shuf; do
    run MAX_ITERS=1 SAVE_SOLUTION=0 "$V1" "$m.mtx"; read -r _ after <<< "$(bw "$OUT")"
    st=$(awk -F, -v m="$m" '$1 == m && $4 == "rcm" {print $5}' study.csv)
    [ "$st" = "$after" ] || { ok=0; msg="$m: cpu_sparse $after, rcm_study $st"; }
done
[ $ok = 1 ] && pass "rcm_study's RCM gives the same bandwidth as cpu_sparse's on 5 matrices" || fail "rcm_study vs cpu_sparse RCM" "$msg"

section "real and structural matrices with Jacobi"
for p in "$V1" "$V2"; do
    for m in bcsstk01 nos4 494_bus el4_shuf p3d; do
        run PRECOND=jacobi MAX_ITERS=20000 "$p" "$m.mtx"
        check "$m.mtx" "$(sol "$p" "$m")"
        if lt "$(resid "$OUT")" 1e-5 && [ $CRC -eq 0 ]; then
            pass "$(basename "$p") $m: converged in $(iters "$OUT") iterations, residual rechecked from file"
        else fail "$(basename "$p") $m" "resid=$(resid "$OUT"); $(tail -1 <<< "$CHK")"; fi
    done
done

section "inputs that must not crash"
for p in "$DENSE" "$V1" "$V2"; do
    run "$p" young1c.mtx.gz
    [ $RC -eq 0 ] && grep -q SKIPPED <<< "$OUT" && pass "$(basename "$p"): complex-valued matrix skipped" || fail "$(basename "$p"): complex" "$OUT"
    run "$p" rect.mtx
    [ $RC -eq 0 ] && grep -q SKIPPED <<< "$OUT" && pass "$(basename "$p"): non-square matrix skipped" || fail "$(basename "$p"): non-square" "$OUT"
    run "$p" p2dgz.mtx.gz; a=$(iters "$OUT")
    run "$p" p2d.mtx
    [ -n "$a" ] && [ "$a" = "$(iters "$OUT")" ] && pass "$(basename "$p"): .mtx.gz gives the same result as .mtx" || fail "$(basename "$p"): gz" "$a vs $(iters "$OUT")"
done
if [ -f west0167.mtx ]; then
    for p in "$V1" "$V2"; do
        run "$p" west0167.mtx
        [ $RC -eq 0 ] && grep -q "Relative Residual" <<< "$OUT" && pass "$(basename "$p"): unsymmetric matrix runs to the end" \
                                                       || fail "$(basename "$p"): unsymmetric" "rc=$RC"
    done
fi
for p in "$V1" "$V2"; do
    run MAX_ITERS=5 "$p" p2d.mtx
    [ "$(iters "$OUT")" = 5 ] && ! lt "$(resid "$OUT")" 1e-5 && pass "$(basename "$p"): MAX_ITERS cap reported as 5 iterations, not converged" \
                                                       || fail "$(basename "$p"): MAX_ITERS" "iters=$(iters "$OUT")"
done

printf "\n%d passed, %d failed\n" "$PASSED" "$FAILED"
[ "$FAILED" -eq 0 ]
