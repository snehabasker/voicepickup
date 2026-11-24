#!/usr/bin/env bash
# Download the data into $DATA_ROOT (default ~/data/voicepickup):
#   * VoiceBank-DEMAND (Valentini-Botinhao et al., 2017, CC BY 4.0), Edinburgh DataShare
#   * DEMAND noise corpus, 16 kHz (Thiemann et al., 2013, CC BY-SA 3.0), Zenodo
#   * DNSMOS P.835 ONNX model (Microsoft DNS Challenge, CC BY 4.0)
# Idempotent: finished items are skipped, partial downloads resume (curl -C -).
# The Edinburgh server is slow (often < 1 MB/s), so the two hosts download in parallel.
# The official *noisy* train set (~2.5 GB) is only needed for --official training; set
# WITH_NOISY_TRAIN=1 to fetch it. Training normally mixes noise on the fly from clean speech.
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-$HOME/data/voicepickup}"
RAW="$DATA_ROOT/raw"
mkdir -p "$RAW/vbd" "$RAW/demand" "$DATA_ROOT/models"
VBD=https://datashare.ed.ac.uk/bitstream/handle/10283/2791
ZEN=https://zenodo.org/api/records/1227121/files

extract() {  # zip dest [flat]  (python zipfile: no dependency on the unzip binary)
  python3 - "$@" <<'PY'
import os, sys, zipfile
zf, dest, flat = sys.argv[1], sys.argv[2], len(sys.argv) > 3
with zipfile.ZipFile(zf) as z:
    for m in z.infolist():
        if m.is_dir() or "__MACOSX" in m.filename:
            continue
        out = os.path.join(dest, os.path.basename(m.filename) if flat else m.filename)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(m) as src, open(out + ".part", "wb") as dst:
            dst.write(src.read())
        os.replace(out + ".part", out)
PY
}

fetch_zip() {  # url zip dest_dir marker [flat]
  local url=$1 zip=$2 dest=$3 marker=$4; shift 4
  [ -e "$marker" ] && return 0
  curl -fL --retry 20 --retry-delay 5 --retry-all-errors -C - -sS -o "$zip" "$url"
  extract "$zip" "$dest" "$@" && rm "$zip"
}

vbd() {
  # small files first so evaluation can start early
  fetch_zip "$VBD/logfiles.zip" "$RAW/vbd/logfiles.zip" "$RAW/vbd" "$RAW/vbd/log_testset.txt" flat
  sets="clean_testset_wav noisy_testset_wav clean_trainset_28spk_wav"
  [ "${WITH_NOISY_TRAIN:-0}" = 1 ] && sets="$sets noisy_trainset_28spk_wav"
  for f in $sets; do
    fetch_zip "$VBD/$f.zip" "$RAW/vbd/$f.zip" "$RAW/vbd" "$RAW/vbd/$f"
    echo "vbd: $f ok"
  done
}

demand() {
  for n in DLIVING OOFFICE PCAFETER SPSQUARE TBUS \
           DKITCHEN DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING PRESTO PSTATION STRAFFIC TCAR TMETRO; do
    fetch_zip "$ZEN/${n}_16k.zip/content" "$RAW/demand/${n}_16k.zip" "$RAW/demand" "$RAW/demand/$n"
    echo "demand: $n ok"
  done
  local m="$DATA_ROOT/models/sig_bak_ovr.onnx"
  [ -e "$m" ] || curl -fL --retry 20 --retry-all-errors -sS -o "$m" \
    https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
  echo "dnsmos model ok"
}

vbd & p1=$!
demand & p2=$!
set +e
wait $p1; s1=$?
wait $p2; s2=$?
[ $s1 -eq 0 ] && [ $s2 -eq 0 ] || { echo "FAILED (vbd=$s1 demand=$s2)"; exit 1; }
echo "done: $(du -sh "$DATA_ROOT" | cut -f1) in $DATA_ROOT"
