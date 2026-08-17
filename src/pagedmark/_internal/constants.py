"""Registries shared by metadata extraction and provenance classification."""

from __future__ import annotations

from dataclasses import dataclass


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(value.split("|"))


SUPPORTED_FORMATS = frozenset(_tokens(".png|.jpg|.jpeg|.webp|.heic|.heif|.avif"))
AI_METADATA_KEYS = _tokens(
    "parameters|postprocessing|extras|workflow|prompt|Dream|SD:mode|StableDiffusionVersion|"
    "generation_time|Model|Model hash|Seed"
)
AI_KEYWORDS = _tokens(
    "prompt|negative_prompt|sampler|cfg_scale|lora|diffusion|comfy|midjourney|dall-e|dalle|imagen|firefly|c2pa|chatgpt|gpt-4|sora|openai|truepic|stable_diffusion|invokeai"
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
C2PA_CHUNK_TYPE = b"caBX"
PNG_METADATA_CHUNKS = frozenset({b"tEXt", b"iTXt", b"zTXt", b"eXIf", b"iCCP"})
RIFF_METADATA_CHUNKS = frozenset({b"EXIF", b"XMP ", b"ICCP", b"C2PA"})
RIFF_CODED_IMAGE_CHUNKS = frozenset({b"VP8 ", b"VP8L", b"ALPH", b"ANMF"})
C2PA_SIGNATURES = tuple(
    token.encode() for token in _tokens("c2pa|C2PA|jumb|jumd|JUMBF|jumbf|cbor|contentcreds|digid|assertions|manifest")
)


@dataclass(frozen=True, slots=True)
class C2paAiVendor:
    """One issuer signature and its normalized product attribution."""

    issuer: bytes
    org: str
    platform: str | None
    needle: str | None
    synthid: bool = False
    asserts_ai: bool = False
    synthid_requires_watermark_action: bool = False


def _vendor(
    issuer: bytes | str,
    org: str,
    platform: str | None,
    needle: str | None,
    *,
    synthid: bool = False,
    asserts_ai: bool = False,
    synthid_requires_watermark_action: bool = False,
) -> C2paAiVendor:
    token = issuer.encode() if isinstance(issuer, str) else issuer
    return C2paAiVendor(
        token,
        org,
        platform,
        needle,
        synthid=synthid,
        asserts_ai=asserts_ai,
        synthid_requires_watermark_action=synthid_requires_watermark_action,
    )


# Order is product priority when a manifest mentions more than one organization.
C2PA_AI_VENDORS: tuple[C2paAiVendor, ...] = (
    _vendor(b"Microsoft", "Microsoft", "Microsoft (Bing Image Creator / Designer)", "Microsoft"),
    _vendor(b"Adobe", "Adobe", "Adobe Firefly", "Adobe"),
    _vendor(
        b"OpenAI",
        "OpenAI",
        "OpenAI (ChatGPT / gpt-image / DALL-E / Sora)",
        "OpenAI",
        synthid=True,
        synthid_requires_watermark_action=True,
    ),
    _vendor(b"Google", "Google LLC", "Google (Gemini / Imagen)", "Google", synthid=True),
    _vendor(b"Stability AI", "Stability AI", "Stability AI (Stable Image / DreamStudio)", "Stability AI"),
    _vendor(b"Black Forest Labs", "Black Forest Labs", "Black Forest Labs (FLUX)", "Black Forest Labs"),
    _vendor(b"volcengine", "ByteDance (Volcano Engine)", "ByteDance (Doubao / Jimeng / Volcano Engine)", "ByteDance"),
    _vendor(
        "北京火山引擎科技有限公司",
        "ByteDance (Volcano Engine)",
        "ByteDance (Doubao / Jimeng / Volcano Engine)",
        "ByteDance",
    ),
    _vendor(b"Byteplus", "BytePlus (ByteDance)", "ByteDance (Doubao / Jimeng / Volcano Engine)", "ByteDance"),
    _vendor(
        b"Dreamina",
        "ByteDance (Dreamina)",
        "ByteDance (Doubao / Jimeng / Volcano Engine)",
        "ByteDance",
        asserts_ai=True,
    ),
    _vendor(b"Canva", "Canva", "Canva (Magic Media)", "Canva"),
    _vendor(b"Eleven Labs", "ElevenLabs", "ElevenLabs", "ElevenLabs"),
    _vendor(b"fal-ai", "fal.ai", "fal.ai", "fal.ai", asserts_ai=True),
    _vendor(b"Bria", "Bria Artificial Intelligence", "Bria AI", "Bria", asserts_ai=True),
    _vendor(b"Truepic", "Truepic", None, None),
)

C2PA_ISSUERS = {vendor.issuer: vendor.org for vendor in C2PA_AI_VENDORS}
C2PA_IDENTITY_AI_ORGS = frozenset(vendor.org for vendor in C2PA_AI_VENDORS if vendor.asserts_ai)

# Product-specific claim generators can sign through a different upstream issuer.
# Keep this attribution beside the issuer registry so every C2PA consumer has one
# canonical source rather than maintaining a derived product map in identify.py.
C2PA_CLAIM_GENERATOR_PLATFORMS: tuple[tuple[str, str], ...] = (
    ("adobe_firefly", "Adobe Firefly"),
    ("firefly", "Adobe Firefly"),
    ("dreamina", "ByteDance (Doubao / Jimeng / Volcano Engine)"),
    ("higgsfield ai", "Higgsfield AI"),
    ("topaz labs image api", "Topaz Labs"),
    ("tiktok ad creative toolbox", "TikTok Ad Creative Toolbox"),
)

C2PA_AI_TOOLS = {
    token.encode(): label
    for token, label in (
        ("GPT-4o", "GPT-4o"),
        ("ChatGPT", "ChatGPT"),
        ("Sora", "Sora"),
        ("DALL-E", "DALL-E"),
        ("DALL", "DALL-E"),
        ("Imagen", "Imagen"),
        ("Firefly", "Firefly"),
        ("Dreamina", "Dreamina"),
    )
}

C2PA_SOFT_BINDINGS = {
    b"com.adobe.trustmark": "Adobe TrustMark",
    b"com.adobe.icn": "Adobe (content fingerprint)",
    b"com.digimarc": "Digimarc",
    b"com.imatag.lamark": "Imatag (Lamark)",
    b"ai.steg": "Steg.AI",
    b"com.microsoft.invismark": "Microsoft InvisMark",
    b"com.microsoft.wavmark": "Microsoft WavMark",
    b"com.verimatrix": "Verimatrix",
    b"com.nagra.nexguard": "NAGRA NexGuard",
    b"com.aiwatermark": "AIWatermark (Meta PixelSeal)",
    b"ai.trufo": "Trufo",
    b"app.overlai": "Overlai",
    b"com.markany": "MarkAny",
    b"com.mentaport": "Mentaport",
    b"es.lumatrace": "LumaTrace",
    b"ai.verda": "VerdaAI",
    b"ai.contentlens": "ContentLens",
    b"io.iscc": "ISCC (content code)",
}

AI_GENERATOR_TOKENS = frozenset(
    {
        "firefly",
        "dall-e",
        "dalle",
        "midjourney",
        "stable diffusion",
        "stable-diffusion",
        "stablediffusion",
        "comfyui",
        "automatic1111",
        "invokeai",
        "imagen",
        "gpt-image",
        "nightcafe",
        "ideogram",
        "leonardo",
        "flux",
        "dreamstudio",
        "novelai",
        "reve.com",
        "aphrodite ai",
        "apple photos clean up",
        "fal-ai",
    }
)

_C2PA_ACTION_NAMES = _tokens("created|converted|edited|filtered|cropped|resized|opened|placed|watermarked.unbound")
C2PA_ACTIONS = {f"c2pa.{action}".encode(): action for action in _C2PA_ACTION_NAMES}


# TC260 producer identity -> the mark key whose vendor signs with it now lives on the
# registry rows (``KnownMark.tc260_producer_codes``, read through
# ``watermark_registry.tc260_producer_vendors``). Keeping the codes beside the mark is
# what stops a newly registered TC260 vendor from silently falling back to ByteDance.
#
# What a TC260 label confirms when its producer is absent or unmapped. Historical
# behaviour, kept as the fallback so an unrecognized producer never regresses to no
# relaxation at all: ByteDance's two products are the ones the relaxed band was
# calibrated on (see _text_mark_engine._DEFAULT_PROVENANCE_NCC_FACTOR).
TC260_FALLBACK_VENDORS: frozenset[str] = frozenset({"doubao", "jimeng"})
