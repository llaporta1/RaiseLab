# Private-PGM / MBI attribution

`scripts/prototype2_pgm_backend.py` contains compression/decompression and
tree-selection routines adapted from `mechanisms/mst.py` in the official
Private-PGM / MBI project:

- Project: https://github.com/ryan112358/mbi
- Source revision: `5f97065a17b3e53cfefc7a8ce3e293fd153dd064`
- Original file: https://github.com/ryan112358/mbi/blob/5f97065a17b3e53cfefc7a8ce3e293fd153dd064/mechanisms/mst.py
- License: Apache License, Version 2.0, included in `MBI-Apache-2.0.txt`

The local adaptation replaces legacy `Domain.attrs` data conversions with
the current MBI domain/data API. It also exposes the selection inference
iteration count for explicitly labeled software smoke tests, uses the verified
upstream module's dependencies, and returns the privately selected edges to
the retaining adapter. The compression thresholds, weighted measurements,
decompression probabilities, normal iteration counts, scoring, budget
allocation, edge selection, and random-number call order follow the pinned
source.

The upstream source file has no separate copyright header, and the pinned
repository has no `NOTICE` file. This attribution documents the source and
local changes; it is not an upstream endorsement. The repository's existing
MIT license does not replace the Apache license on this adapted code.
