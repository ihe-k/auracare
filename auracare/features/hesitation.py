"""
Hesitation measures: word retrieval latency and unfilled pause duration.

Both exist because pause ratio alone is a poor escalation trigger. Calibration
against the practice recordings put the median silent fraction of ordinary
connected speech near 48 %, so a single threshold on that ratio fires
constantly on healthy speakers. These two measures give the escalation rule
something to corroborate a high ratio with.

**Word retrieval latency** is the silence immediately before a content word -
a noun, verb, adjective or adverb. That is where lexical access happens.
Pausing before "the" is phrasing; pausing before "hospital" is retrieval. It
needs word-level alignment, which is why the recogniser is asked for word
timestamps.

**Unfilled pause duration** separates silence from filled hesitation. "Um" and
"uh" are sound, so an energy-based detector already counts them as speech -
but a hesitation usually looks like silence, filler, silence, which the
detector sees as two short pauses rather than one long hesitation. Those are
merged here and marked filled, so what remains is genuine silence. The
distinction matters clinically: a filled pause means the speaker is holding
the floor while thinking, an unfilled one can mean the thread was lost.

Known limit, worth stating: the recogniser transcribes fillers inconsistently
and sometimes drops them entirely. Filled-pause counts are therefore a floor,
not an exact tally, and a missed filler shifts a pause into the unfilled
bucket rather than out of it. That direction is the safe one - it can make the
measure look worse than reality, never better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Segment = Tuple[float, float]

# Fillers, as the recogniser tends to spell them. Kept deliberately wide:
# a false positive here only moves a pause into the filled bucket, which is
# the conservative direction for an escalation rule.
FILLED_PAUSE_TOKENS = frozenset(
    {
        "um", "umm", "ummm", "uhm", "uhmm",
        "uh", "uhh", "uhhh",
        "er", "err", "erm", "ermm",
        "ah", "ahh", "eh", "ehh",
        "hmm", "hm", "hmmm", "mm", "mmm", "mhm",
        "huh",
    }
)

# Closed-class English. Everything not in here (and not a filler) counts as a
# content word. A stoplist is used rather than a part-of-speech tagger on
# purpose: a tagger is another heavy dependency, and for this measure the
# open/closed class split is what matters, not fine-grained tags.
FUNCTION_WORDS = frozenset(
    {
        # determiners and quantifiers
        "a", "an", "the", "this", "that", "these", "those", "each", "every",
        "some", "any", "no", "all", "both", "half", "either", "neither",
        "much", "many", "more", "most", "few", "fewer", "less", "least",
        "several", "enough", "such", "another", "other",
        # pronouns
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
        "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "who", "whom", "whose", "which", "what", "whatever",
        "whoever", "one", "ones",
        # auxiliaries, modals and copula
        "am", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "done", "have", "has", "had", "having",
        "will", "would", "shall", "should", "can", "could", "may", "might",
        "must", "ought", "need", "dare",
        # prepositions and particles
        "about", "above", "across", "after", "against", "along", "among",
        "around", "as", "at", "before", "behind", "below", "beneath",
        "beside", "between", "beyond", "by", "despite", "down", "during",
        "except", "for", "from", "in", "inside", "into", "like", "near",
        "of", "off", "on", "onto", "out", "outside", "over", "past",
        "since", "through", "throughout", "to", "toward", "towards",
        "under", "until", "up", "upon", "with", "within", "without",
        # conjunctions and connectives
        "and", "but", "or", "nor", "so", "yet", "because", "although",
        "though", "while", "whereas", "if", "unless", "whether", "than",
        "then", "when", "where", "why", "how", "however", "therefore",
        # common adverbial fillers that are not lexical retrieval
        "not", "just", "very", "too", "also", "only", "even", "still",
        "there", "here", "now", "well", "okay", "ok", "yes", "yeah", "no",
        "please", "thank", "thanks",
    }
)

# A gap below this is articulation, not retrieval: normal coarticulation and
# alignment jitter sit in this range.
DEFAULT_MIN_GAP_S = 0.05
# A gap at or above this counts as an overt hesitation event.
DEFAULT_HESITATION_S = 0.25
# How close a filler has to sit to a pause edge to be treated as filling it.
DEFAULT_FILLER_TOLERANCE_S = 0.30


def is_filler(token: str) -> bool:
    return token.strip().lower().strip(".,!?;:-") in FILLED_PAUSE_TOKENS


def is_content_word(token: str) -> bool:
    """True for open-class words: the ones lexical retrieval has to find."""
    clean = token.strip().lower().strip(".,!?;:-'\"")
    if not clean:
        return False
    if clean in FILLED_PAUSE_TOKENS or clean in FUNCTION_WORDS:
        return False
    # Require at least one letter, so digits and stray punctuation tokens do
    # not register as content.
    return any(ch.isalpha() for ch in clean)


@dataclass
class RetrievalResult:
    """Word retrieval latency over one recording."""

    mean_latency_s: Optional[float] = None
    max_latency_s: Optional[float] = None
    n_content_words: int = 0
    n_hesitations: int = 0
    hesitation_rate: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        def r(v, nd=3):
            return None if v is None else round(float(v), nd)

        return {
            "mean_latency_s": r(self.mean_latency_s),
            "max_latency_s": r(self.max_latency_s),
            "n_content_words": self.n_content_words,
            "n_hesitations": self.n_hesitations,
            "hesitation_rate": r(self.hesitation_rate, 4),
            "notes": list(self.notes),
        }


@dataclass
class PauseBreakdown:
    """Silent pauses split from filled ones."""

    mean_unfilled_s: Optional[float] = None
    longest_unfilled_s: Optional[float] = None
    n_unfilled: int = 0
    n_filled: int = 0
    unfilled_segments: List[Segment] = field(default_factory=list)
    filled_segments: List[Segment] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        def r(v, nd=3):
            return None if v is None else round(float(v), nd)

        return {
            "mean_unfilled_s": r(self.mean_unfilled_s),
            "longest_unfilled_s": r(self.longest_unfilled_s),
            "n_unfilled": self.n_unfilled,
            "n_filled": self.n_filled,
            "notes": list(self.notes),
        }


def word_retrieval_latency(
    words: Sequence,
    min_gap_s: float = DEFAULT_MIN_GAP_S,
    hesitation_s: float = DEFAULT_HESITATION_S,
) -> RetrievalResult:
    """
    Mean silence immediately before each content word.

    `words` is a sequence of objects with `.text`, `.start` and `.end` - the
    recogniser's word timings. Fillers are skipped as targets (hesitating
    before "um" is not retrieval) but still close the gap for the word after
    them, because the speaker was audibly occupied during the filler.
    """
    out = RetrievalResult()

    if len(words) < 2:
        out.notes.append(
            "word timings unavailable or too short for a retrieval estimate"
        )
        return out

    latencies: List[float] = []
    for i in range(1, len(words)):
        current = words[i]
        text = getattr(current, "text", "") or ""
        if not is_content_word(text):
            continue

        previous = words[i - 1]
        gap = float(getattr(current, "start", 0.0)) - float(
            getattr(previous, "end", 0.0)
        )
        # Alignment can overlap adjacent words slightly; a negative gap is a
        # timing artefact, not a negative latency.
        gap = max(0.0, gap)
        out.n_content_words += 1
        if gap < min_gap_s:
            latencies.append(0.0)
            continue
        latencies.append(gap)
        if gap >= hesitation_s:
            out.n_hesitations += 1

    if not latencies:
        out.notes.append("no content words found after the first word")
        return out

    out.mean_latency_s = float(sum(latencies) / len(latencies))
    out.max_latency_s = float(max(latencies))
    out.hesitation_rate = out.n_hesitations / len(latencies)
    return out


def _overlaps(a: Segment, b: Segment, tolerance: float) -> bool:
    return a[0] - tolerance <= b[1] and a[1] + tolerance >= b[0]


def split_pauses(
    pause_segments: Sequence[Segment],
    words: Sequence,
    tolerance_s: float = DEFAULT_FILLER_TOLERANCE_S,
) -> PauseBreakdown:
    """
    Split pauses into genuinely silent ones and filled hesitations.

    Two pauses separated only by a filler are merged into a single filled
    span, because that whole stretch is one hesitation rather than two short
    gaps with a word between them.
    """
    out = PauseBreakdown()

    pauses = sorted((float(a), float(b)) for a, b in pause_segments)
    if not pauses:
        return out

    fillers: List[Segment] = [
        (float(getattr(w, "start", 0.0)), float(getattr(w, "end", 0.0)))
        for w in words
        if is_filler(getattr(w, "text", "") or "")
    ]

    if not words:
        # No alignment at all: every pause is reported as unfilled, and the
        # caller is told why, rather than silently implying fillers were
        # checked for and none found.
        out.notes.append(
            "no word timings available, so filled pauses could not be "
            "identified; all pauses are reported as unfilled"
        )
    elif not fillers:
        # Whisper is trained to produce clean, readable transcripts and drops
        # disfluencies: measured on a 10-minute spontaneous recording it
        # returned 0 fillers across 606 words. A filled-pause count of zero
        # therefore says nothing about whether the speaker produced any, and
        # must not be read as "this patient did not hesitate audibly".
        # The consequence is conservative - a silence/filler/silence
        # hesitation stays split into two shorter silences rather than being
        # merged into one long one, which understates severity rather than
        # overstating it.
        out.notes.append(
            "the recogniser returned no filled pauses; it strips disfluencies "
            "from transcripts, so this is a limit of the transcription rather "
            "than evidence that the speaker produced none"
        )

    merged: List[Tuple[Segment, bool]] = []
    i = 0
    while i < len(pauses):
        start, end = pauses[i]
        filled = any(_overlaps((start, end), f, tolerance_s) for f in fillers)

        # Absorb following pauses while a filler, and nothing else, bridges
        # the gap between them. If real speech occurs in between, the two
        # silences are separate events and merging them would invent one long
        # pause that the speaker never made.
        while i + 1 < len(pauses):
            next_start, next_end = pauses[i + 1]
            lo, hi = end - tolerance_s, next_start + tolerance_s
            in_bridge = [
                wd
                for wd in words
                if float(getattr(wd, "start", 0.0)) >= lo
                and float(getattr(wd, "end", 0.0)) <= hi
            ]
            if not in_bridge:
                break
            if not all(is_filler(getattr(wd, "text", "") or "") for wd in in_bridge):
                break
            end = next_end
            filled = True
            i += 1

        merged.append(((start, end), filled))
        i += 1

    for seg, filled in merged:
        if filled:
            out.filled_segments.append(seg)
        else:
            out.unfilled_segments.append(seg)

    out.n_filled = len(out.filled_segments)
    out.n_unfilled = len(out.unfilled_segments)

    if out.unfilled_segments:
        durations = [b - a for a, b in out.unfilled_segments]
        out.mean_unfilled_s = float(sum(durations) / len(durations))
        out.longest_unfilled_s = float(max(durations))

    return out


__all__ = [
    "FILLED_PAUSE_TOKENS",
    "FUNCTION_WORDS",
    "PauseBreakdown",
    "RetrievalResult",
    "is_content_word",
    "is_filler",
    "split_pauses",
    "word_retrieval_latency",
]
