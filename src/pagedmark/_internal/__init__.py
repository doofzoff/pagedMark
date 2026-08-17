"""Private namespace for metadata parsing and regeneration internals.

Deliberately empty. It carried a PEP 562 ``__getattr__`` re-exporting
``WatermarkRemover`` and ``remove_ai_metadata`` as a "compatibility namespace",
but nothing ever reached for either through this package -- every caller imports
the submodule directly. The laziness it defended is real and still enforced, just
elsewhere: importing a light submodule (``_internal.c2pa`` / ``_internal.constants``
from ``identify``) must not pull ``watermark_remover``, which imports torch at
module top. That property comes from those direct submodule imports, not from a
shim here; a re-export in this file would be the one thing that could break it.

Keep this module free of imports. ``import pagedmark.identify`` stays
around 36 MB even in a full install where torch is present; routing anything heavy
through here inflated it to roughly 420 MB and risked OOM on a 512 MB host.
"""
