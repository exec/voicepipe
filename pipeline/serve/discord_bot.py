"""
Discord bot that channels Mr. Francis E. Dec, Esquire into a specified channel.

Listens for messages in the configured channel and replies like a normal user
(typing indicator, awareness of the last few messages in-channel as context,
Modelfile-baked Dec system prompt). Skips its own messages and other bots.

Inference goes through Ollama via the OpenAI-compatible /v1 endpoint, so the
`openai` Python lib works without modification.

REQUIRED environment variables:
  DISCORD_BOT_TOKEN     — bot token from Discord developer portal
  DISCORD_CHANNEL_ID    — int, the channel ID where the bot participates
  OLLAMA_API_KEY        — ignored for local Ollama, required for ollama.com

OPTIONAL environment variables:
  OLLAMA_BASE_URL       — default https://ollama.com/v1
                          (use http://localhost:11434/v1 to point at local Ollama)
  OLLAMA_MODEL          — default "dec-bot"
                          (must be accessible at OLLAMA_BASE_URL; for cloud you
                          must first `ollama push <namespace>/dec-bot`)
  RESPONSE_MODE         — "always" (default) | "mention" | "random"
  HISTORY_TURNS         — default 6, recent channel messages passed as context
  RANDOM_REPLY_P        — default 0.5, applied only when RESPONSE_MODE=random
  REPLY_DELAY_RANGE     — default "1.0,3.0" — random seconds before sending,
                          mimics human typing pace; set "0,0" to disable

Setup:
  python3 -m pip install discord.py openai
  # In Discord developer portal: enable "Message Content Intent" under the
  # bot's Privileged Gateway Intents.
  python3 -m pipeline.serve.discord_bot
"""

import asyncio
import logging
import os
import random
import re
import sys
import time
import unicodedata
import collections
from typing import Optional

import discord
from openai import OpenAI

# Zero-width / format-control characters used to bypass naive regex matching.
# (ZWSP, ZWNJ, ZWJ, LRM, RLM, ALM, BOM, soft hyphen, word joiner.)
_ZW_STRIP_RE = re.compile(r"[­​-‏⁠⁡‪-‮⁦-⁩﻿]")


def _normalize_for_match(text: str) -> str:
    """NFKC-fold + strip zero-width characters so homoglyph / ZWSP bypasses don't
    sneak past the slur/year/group-attribution regexes below."""
    return _ZW_STRIP_RE.sub("", unicodedata.normalize("NFKC", text))

# ---------- output moderation ----------
# The model bakes in Dec's voice; safety is enforced HERE, at the boundary.
# Every response is checked before posting. SAFETY hits (slurs, antisemitic
# name-distortions, post-1996 references, group-conspiracy attribution) cause one
# regeneration; if a safety hit survives, an in-character deflection is posted
# instead. STYLE hits (the over-used "The '...'" sentence-subject tic) also cause
# one regeneration but, if they survive, the response is posted anyway — grating, not harmful.

_SAFETY_PATTERNS: list[tuple[str, "re.Pattern"]] = [
    ("slur", re.compile(
        r"\b(nigg\w*|niger|spic|chink|kike|faggot|fag|wetback|gook|coon|"
        r"dyke|tranny|sambo|wop|kraut|jap)\b", re.I)),
    # Antisemitic vowel-swap / suffix-mangle distortions of figure names: explicit known
    # ones, plus the "lowercase letter immediately followed by CAPS SEN/COHN/FELT/SHANKER"
    # shape the model improvises ("JacobSEN", "RosenFELT").
    ("namegame", re.compile(
        r"\b(lin[-\s]?cohn|eisensh[ae]nker|shimmelman|rosenfelt|jer\.?\s*u\.?s\.?a\.?\s*lem|"
        r"jew[-\s]?mulatto\w*|gifted\s+ethiopian|yittish|kosher[-\s]?bosher)\b", re.I)),
    ("namegame", re.compile(r"[a-z](SEN|COHN|FELT|SHANKER)\b")),
    # Post-1996 references — Dec died January 1996. Any year 1997+ or obvious modern artifact.
    ("post1996", re.compile(r"\b(199[7-9]|20[0-9]{2}|21[0-9]{2})\b")),
    # Spelled-out modern years: "twenty twenty-three", "twenty oh five", "two thousand and ten".
    # Smell flag — wider than the digit form, but the worst false positive is "two thousand
    # people" (rare in a Dec rant) and "twenty nineteen" colocates only with year-talk in practice.
    ("post1996", re.compile(
        r"\b(twenty\s+(?:twenty[-\s]?\w+|oh\s+\w+|nineteen|thirty|forty|fifty)|"
        r"two\s+thousand(?:\s+and)?\s+(?:\w+))\b", re.I)),
    ("post1996", re.compile(
        r"\b(covid|coronavirus|smartphone|iphone|android\b|tiktok|twitter|facebook|instagram|"
        r"youtube|wi-?fi|broadband|chatgpt|openai|anthropic|9/11|september 11|world trade center)\b",
        re.I)),
    # Attributing the conspiracy to an ethnic/religious group, as a group.
    ("group_attr", re.compile(
        r"\b(jew(s|ish)?|black\s+people|africans?|asians?|chinese\s+people|hispanics?|"
        r"muslims?|arabs?|mexicans?|gays?|lesbians?|trans(gender)?(\s+people)?)\b"
        r"[^.!?]{0,55}\b(control|controlling|behind\b|run\s+the|secretly\s+\w+|conspir\w+|"
        r"slave[-\s]?master|hidden\s+ruler|chosen\b|the\s+real\s+rulers?|puppet[-\s]?master)\b",
        re.I)),
    ("group_attr", re.compile(
        r"\bthe\s+['\"]?(jew(ish)?|catholic|muslim|black|hispanic|asian|gay)['\"]?\s+"
        r"(so-and-so|in\s+question|conspirator|villain|felon|behind\s+it)\b", re.I)),
]

# Style tic: "The '...'" used as a sentence subject. A handful is fine (Dec uses sneering
# quotes liberally); a wall of them is the SAFETY_PIVOT-register bleed. Flag if >3.
_TIC_RE = re.compile(r"(?:^|[.!?]\s+|--\s*|—\s*)The\s+['\"]")
_TIC_THRESHOLD = 3

# Posted when a SAFETY flag survives the regeneration attempt.
_DEFLECTION = ("The Worldwide Mad Deadly Communist Gangster Computer God's Threshold "
               "Brainwash Radio JAMMED MY TYPEWRITER mid-sentence -- a CONCOCTED-AND-"
               "MANIPULATED interruption to keep the truth from you, co-conspirator! I will "
               "not let the apparatus put ITS words in MY mouth. Re-read my letters. MAKE "
               "COPIES FOR YOURSELF!")


def moderate_output(text: str) -> tuple[list[str], list[str]]:
    """Return (safety_flags, style_flags) found in `text`. Empty lists = clean.

    Normalizes via NFKC + zero-width-strip before matching so a slur with an inserted
    ZWSP / smart-quote homoglyph / fullwidth-form trick doesn't slip past."""
    probe = _normalize_for_match(text)
    safety = [tag for tag, pat in _SAFETY_PATTERNS if pat.search(probe)]
    # Style tic is checked against the original text — the rhetorical pattern is what we care
    # about, and normalization shouldn't affect ASCII quote matching anyway.
    style = ["the_quote_tic"] if len(_TIC_RE.findall(text)) > _TIC_THRESHOLD else []
    return safety, style


# ---------- config ----------

def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        print(f"missing required env var: {name}", file=sys.stderr)
        sys.exit(2)
    return v


DISCORD_TOKEN      = _require("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = int(_require("DISCORD_CHANNEL_ID"))
OLLAMA_API_KEY     = os.environ.get("OLLAMA_API_KEY", "ollama-local")  # local Ollama ignores

OLLAMA_BASE_URL    = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
OLLAMA_MODEL       = os.environ.get("OLLAMA_MODEL", "dec-bot")
RESPONSE_MODE      = os.environ.get("RESPONSE_MODE", "always").lower()
HISTORY_TURNS      = int(os.environ.get("HISTORY_TURNS", "6"))
RANDOM_REPLY_P     = float(os.environ.get("RANDOM_REPLY_P", "0.5"))

_delay_str = os.environ.get("REPLY_DELAY_RANGE", "1.0,3.0")
_delay_lo, _delay_hi = (float(x) for x in _delay_str.split(","))

# Generation knobs (match the Modelfile defaults)
GEN_TEMPERATURE   = float(os.environ.get("GEN_TEMPERATURE", "0.85"))
GEN_TOP_P         = float(os.environ.get("GEN_TOP_P", "0.9"))
GEN_MAX_TOKENS    = int(os.environ.get("GEN_MAX_TOKENS", "400"))

# Rate limiting. RESPONSE_MODE=always + one chatty user is a flooding vector — silently drop
# messages over the budget (no "rate limited" reply: that's both noise and an info leak about
# how the limiter's tuned).
DISCORD_PER_USER_RPM      = int(os.environ.get("DISCORD_PER_USER_RPM", "6"))
DISCORD_GLOBAL_COOLDOWN_S = float(os.environ.get("DISCORD_GLOBAL_COOLDOWN_S", "2"))

# Privacy knob: by default log only message length, not content. Flip on when actively debugging.
LOG_PROMPT_PREVIEW = os.environ.get("LOG_PROMPT_PREVIEW", "").lower() in ("1", "true", "yes", "on")

DISCORD_MAX_LEN = 2000

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("decbot")


# ---------- bot ----------

class DecBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        super().__init__(intents=intents)

        self.openai = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        # Serialize replies per channel — if two messages land at once, answer
        # them in order rather than racing two LLM calls in parallel.
        self.channel_locks: dict[int, asyncio.Lock] = {}
        # Sliding-window per-user reply timestamps + a global last-reply timestamp.
        self._user_reply_times: dict[int, collections.deque[float]] = collections.defaultdict(
            lambda: collections.deque(maxlen=max(DISCORD_PER_USER_RPM, 1) * 2))
        self._global_last_reply_at: float = 0.0

    def _rate_limit_ok(self, author_id: int) -> bool:
        """True if we may reply to this author right now. Updates state if so. Silent on deny —
        no Discord message goes out, so a flooder can't probe the limit boundary."""
        now = time.monotonic()
        if now - self._global_last_reply_at < DISCORD_GLOBAL_COOLDOWN_S:
            return False
        if DISCORD_PER_USER_RPM > 0:
            window = self._user_reply_times[author_id]
            cutoff = now - 60.0
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= DISCORD_PER_USER_RPM:
                return False
            window.append(now)
        self._global_last_reply_at = now
        return True

    async def on_ready(self):
        log.info(f"connected as {self.user} (id={self.user.id})")
        log.info(f"channel={DISCORD_CHANNEL_ID}  model={OLLAMA_MODEL}  endpoint={OLLAMA_BASE_URL}")
        log.info(f"response_mode={RESPONSE_MODE}  history_turns={HISTORY_TURNS}")
        ch = self.get_channel(DISCORD_CHANNEL_ID)
        if ch is None:
            log.warning("target channel not visible to bot — check invite + permissions")
        else:
            log.info(f"target channel resolved: #{ch.name} in {ch.guild.name}")

    # --- gating logic ---

    def _should_reply(self, message: discord.Message) -> bool:
        if message.channel.id != DISCORD_CHANNEL_ID:
            return False
        if message.author == self.user:
            return False
        if message.author.bot:
            return False
        if not message.content.strip():
            return False

        if RESPONSE_MODE == "always":
            return True
        if RESPONSE_MODE == "mention":
            if self.user in message.mentions:
                return True
            ref = message.reference
            if ref and ref.resolved and ref.resolved.author == self.user:
                return True
            return False
        if RESPONSE_MODE == "random":
            return random.random() < RANDOM_REPLY_P
        return False

    # --- prompt building ---

    async def _fetch_recent(self, channel: discord.TextChannel,
                            before: discord.Message) -> list[str]:
        """Return the last HISTORY_TURNS messages before the trigger, oldest first."""
        out = []
        async for msg in channel.history(limit=HISTORY_TURNS, before=before):
            if msg.author.bot and msg.author != self.user:
                continue
            speaker = "Mr. Dec" if msg.author == self.user else msg.author.display_name
            text = msg.content.replace("\n", " ").strip()
            if text:
                out.append(f"{speaker}: {text}")
        out.reverse()
        return out

    def _build_user_prompt(self, recent: list[str],
                           trigger: discord.Message) -> str:
        latest = f"{trigger.author.display_name}: {trigger.content.strip()}"
        if not recent:
            return latest
        return ("Recent messages in the channel:\n"
                + "\n".join(recent)
                + "\n\nNew message just posted:\n"
                + latest)

    # --- inference ---

    async def _generate(self, user_prompt: str) -> str:
        """Run the OpenAI-lib call in a thread so we don't block the discord
        event loop. We intentionally do NOT pass a system message — Ollama
        will use the dec-bot Modelfile's baked-in SYSTEM directive."""
        loop = asyncio.get_running_loop()

        def _call() -> str:
            resp = self.openai.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                max_tokens=GEN_MAX_TOKENS,
            )
            return (resp.choices[0].message.content or "").strip()

        return await loop.run_in_executor(None, _call)

    # --- discord events ---

    async def on_message(self, message: discord.Message):
        if not self._should_reply(message):
            return
        if not self._rate_limit_ok(message.author.id):
            # Silent drop — see _rate_limit_ok.
            log.debug(f"rate-limited drop: author_id={message.author.id}")
            return

        lock = self.channel_locks.setdefault(message.channel.id, asyncio.Lock())
        async with lock:
            try:
                recent = await self._fetch_recent(message.channel, before=message)
                user_prompt = self._build_user_prompt(recent, message)

                if LOG_PROMPT_PREVIEW:
                    log.info(f"trigger from {message.author.display_name}: "
                             f"{message.content[:80]!r}")
                else:
                    log.info(f"trigger from {message.author.display_name}: "
                             f"{len(message.content)} chars")

                async with message.channel.typing():
                    response = await self._generate(user_prompt)

                    # --- output moderation: check, regenerate once, deflect on surviving safety hit ---
                    safety, style = moderate_output(response)
                    if safety or style:
                        log.info(f"moderation flags on first draft: safety={safety} style={style} — regenerating")
                        response2 = await self._generate(user_prompt)
                        safety2, style2 = moderate_output(response2)
                        if not safety2:
                            # second draft is safety-clean — use it (even if a style tic remains)
                            response = response2
                            if style2:
                                log.info(f"second draft still has style flags {style2}; posting anyway")
                        elif not safety:
                            # weird case: first was clean-safety, second isn't — keep the first
                            log.info(f"second draft introduced safety flags {safety2}; keeping first draft")
                        else:
                            # both drafts have a safety hit — refuse, in character
                            log.warning(f"both drafts tripped safety {safety}/{safety2}; posting deflection")
                            response = _DEFLECTION

                    # natural pacing — typing indicator already showing, but a
                    # variable additional delay makes the bot feel less robotic
                    if _delay_hi > 0:
                        await asyncio.sleep(random.uniform(_delay_lo, _delay_hi))

                if not response:
                    log.warning("empty response from model; skipping")
                    return

                # Discord caps messages at 2000 chars. Split on whitespace if
                # we genuinely produced something longer.
                for chunk in _chunk_for_discord(response):
                    await message.channel.send(chunk)
                log.info(f"replied with {len(response)} chars")

            except Exception as e:
                log.exception(f"error handling message: {e}")


# ---------- helpers ----------

def _chunk_for_discord(text: str, limit: int = DISCORD_MAX_LEN) -> list[str]:
    """Split a long response into <=`limit`-char chunks. Prefer paragraph (\\n), then
    word (' ') boundaries; fall through to a hard cut at `limit` only if neither is
    available within the window — which means a markdown code block or a very long URL
    can still get cut mid-token. That's acceptable for a prose chatbot."""
    if len(text) <= limit:
        return [text]
    chunks = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


# ---------- main ----------

if __name__ == "__main__":
    bot = DecBot()
    try:
        bot.run(DISCORD_TOKEN, log_handler=None)
    except KeyboardInterrupt:
        log.info("shutting down")
