#!/usr/bin/env bash
# Gera assets/demo.gif a partir do quickstart, para o topo do README.
#
# Requisitos (uma vez):
#   brew install asciinema agg        # macOS
#   # ou: sudo apt install asciinema  +  cargo install agg
#   pip install mangaba
#   export GOOGLE_API_KEY="sua-chave"
#
# Uso:
#   bash scripts/record_demo.sh
#   # -> assets/demo.cast + assets/demo.gif
#   # depois descomente o bloco <img> marcado como DEMO-GIF no README.md
set -euo pipefail

cd "$(dirname "$0")/.."

asciinema rec --overwrite --title "Mangaba AI quickstart" assets/demo.cast \
  -c "python examples/quickstart.py"

agg assets/demo.cast assets/demo.gif

echo "OK: assets/demo.gif gerado. Descomente o bloco DEMO-GIF no README.md."
