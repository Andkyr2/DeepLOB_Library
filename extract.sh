set -euo pipefail

for f in data/raw/*.7z; do
    name=$(basename $f ".7z")
    mkdir -p "data/extracted/$name"
    7z x $f -o"data/extracted/$name" -y
done