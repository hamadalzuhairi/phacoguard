#!/usr/bin/env bash
# Rule 4.3: purge any temporary files that could contain institutional data. Run inside Environment B at the end.
set -euo pipefail
for d in /tmp/phacoguard outputs/institutional /nvme/phacoguard/institutional; do [ -d "$d" ] && rm -rf "$d" && echo "purged $d"; done
echo "purge complete; confirm with the Data Protection Officer"
