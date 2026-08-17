# Code provenance

This page records notices required by source dependencies and licensed derivative work.

## Licensed derivative work

- The DWT-DCT implementation derives from ShieldMnt's
  [`invisible-watermark`](https://github.com/ShieldMnt/invisible-watermark), licensed
  under MIT. Its notice ships in `src/pagedmark/licenses/invisible-watermark-MIT.txt`.
  The decode path is a vectorized reformulation, not a transcription: it produces
  the same bits and is checked against upstream's own decoder, but it no longer
  corresponds line by line to `imwatermark/maxDct.py`. Diffing the two files will
  show structurally different code, which is expected and does not mean the
  derivation notice is stale.

## Licensed test fixtures

- `data/fixtures/provenance/adobe-trustmark-p.png` is Adobe's official
  TrustMark Variant P example, licensed under MIT. Its source commit, digest,
  and reproduced license are recorded beside the fixture in
  `data/fixtures/README.md` and `data/licenses/adobe-trustmark-MIT.txt`.
