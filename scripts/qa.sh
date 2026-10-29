#!/usr/bin/env bash
# QA autonomo: no instala paquetes ni inicia el experimento oficial.
set -uo pipefail
script_dir="${BASH_SOURCE[0]%/*}"
if [[ "$script_dir" == "${BASH_SOURCE[0]}" ]]; then script_dir="."; fi
cd -- "$script_dir/.." || exit 2
# Local shell bootstrap; no policy changes, remote agents or automatic approvals.
trace_id="$(date -u +%Y%m%dT%H%M%SZ)_${BOTRYTIS_PC_ID:-PC_NO_DECLARADA}_qa_shell_$$_${RANDOM}_${RANDOM}"
trace_dir="runs/v4/_shell/$trace_id"
mkdir -p -- "$trace_dir" || exit 3
trace_events="$trace_dir/events.jsonl"
printf '{"event":"bootstrap_start","run_id":"%s","status":"QA_MANUAL_PENDIENTE"}\n' "$trace_id" >> "$trace_events"
exec > >(tee -a "$trace_dir/execution.log") 2>&1
finish_trace() {
  trace_code=$?
  printf '{"event":"bootstrap_end","run_id":"%s","returncode":%s,"status":"QA_MANUAL_PENDIENTE"}\n' "$trace_id" "$trace_code" >> "$trace_events"
}
trap finish_trace EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
pycmd=()
if [[ -n "${BOTRYTIS_PYTHON:-}" ]]; then
  pycmd=("$BOTRYTIS_PYTHON")
elif [[ -f .venv/Scripts/python.exe ]]; then
  pycmd=("./.venv/Scripts/python.exe")
elif [[ -f .venv/bin/python ]]; then
  pycmd=("./.venv/bin/python")
else
  for candidate in python.exe python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3,10))' >/dev/null 2>&1; then
      pycmd=("$candidate"); break
    fi
  done
  if [[ ${#pycmd[@]} -eq 0 ]] && command -v py.exe >/dev/null 2>&1; then
    pycmd=(py.exe -3)
  fi
fi
if [[ ${#pycmd[@]} -eq 0 ]]; then
  printf '%s\n' 'BLOCKED: Python >=3.10 no encontrado. Use scripts/Invoke-QA.ps1 para conservar el diagnostico de arranque.'
  if command -v mkdir >/dev/null 2>&1; then
    mkdir -p logs_qa
    printf '{"status":"BLOCKED","reason":"Python >=3.10 no disponible","bash_version":"%s"}\n' "$BASH_VERSION" > "logs_qa/bootstrap_$$_$RANDOM.json"
  fi
  exit 3
fi
"${pycmd[@]}" tools/qa.py --bash-version "$BASH_VERSION" "$@"
exit $?
