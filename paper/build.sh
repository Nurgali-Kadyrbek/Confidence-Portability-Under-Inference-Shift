#!/usr/bin/env bash
# Compile a manuscript with the official MDPI class, installing any TeX Live
# package a missing file belongs to. TinyTeX starts from a minimal scheme, so
# the dependency set is discovered from the log rather than guessed.
set -uo pipefail
export PATH="$HOME/envs/tools/bin:$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
cd "$(dirname "$0")"
DOC="${1:-cpis}"
BIB="${2:-yes}"

resolve_missing() {
  local log="$1" changed=0 file pkg base
  for file in $(grep -oE "File \`[^']+' not found" "$log" | sed "s/File \`//;s/' not found//" | sort -u); do
    base="${file%.*}"
    # The package name usually equals the file basename; fall back to a file
    # search in the TeX Live database when it does not.
    if tlmgr install "$base" >/dev/null 2>&1 && kpsewhich "$file" >/dev/null 2>&1; then
      echo "  installed $base (for $file)"
      changed=1
      continue
    fi
    pkg=$(tlmgr search --global --file "/$file" 2>/dev/null | awk -F: '/^[^ \t].*:$/{print $1; exit}')
    if [ -n "$pkg" ]; then
      echo "  installing $pkg (for $file)"
      tlmgr install "$pkg" >/dev/null 2>&1 && changed=1
    else
      echo "  UNRESOLVED: $file"
    fi
  done
  return $((1 - changed))
}

for attempt in $(seq 1 25); do
  pdflatex -interaction=nonstopmode -file-line-error "$DOC.tex" >"$DOC.log" 2>&1
  if grep -qE "File \`[^']+' not found" "$DOC.log"; then
    echo "attempt $attempt: resolving missing files"
    resolve_missing "$DOC.log" || { echo "no progress; stopping"; break; }
    continue
  fi
  break
done

if [ "$BIB" = "yes" ] && [ -f "$DOC.aux" ]; then
  bibtex "$DOC" >"$DOC.bibtex.log" 2>&1 || true
fi
for i in 1 2 3; do
  pdflatex -interaction=nonstopmode -file-line-error "$DOC.tex" >"$DOC.log" 2>&1
done
if grep -qE '(^! |Package .* Error:|Undefined control sequence|Emergency stop|Fatal error)' "$DOC.log"; then
  echo "FAILED: LaTeX reported an error"
  grep -E '(^! |Package .* Error:|Undefined control sequence|Emergency stop|Fatal error)' "$DOC.log" | head -20
  exit 1
fi
if [ -f "$DOC.pdf" ]; then
  echo "OK: $DOC.pdf ($(stat -c %s "$DOC.pdf") bytes, $(pdfinfo "$DOC.pdf" 2>/dev/null | awk '/^Pages/{print $2}') pages)"
else
  echo "FAILED to produce $DOC.pdf"
  grep -E "^\./|^! |Error" "$DOC.log" | head -20
  exit 1
fi
