"""
Phase 8b - Plain-English risk reports.

SHAP answers "which features moved this prediction, and by how much, in
log-odds". That is the right answer for a data scientist and the wrong answer
for everyone else. This module turns those numbers into sentences a person can
act on.

Two narrators, same input:

* TemplateNarrator - deterministic, rule-based, no network, no API key. Always
  available, and always the fallback.
* ClaudeNarrator   - sends the same structured facts to Claude and asks for a
  fluent summary. Produces better prose; requires ANTHROPIC_API_KEY.

The important design decision is that **the LLM is never given the raw model or
asked to assess risk itself**. It receives a fixed set of already-computed
facts and is instructed to rephrase them. The numbers come from SHAP; the
language model only writes the sentences. That keeps the medical claims
attributable to the model and the data, not to an LLM's own opinion, and means
a failed API call degrades to the template rather than to nonsense.
"""

import os
import textwrap

import config

# Google Gemini is the default provider: its free tier needs no billing setup,
# which matters for a project meant to be reproducible by anyone who clones it.
#
# Two reasons this is a "-latest" alias rather than a pinned version: Google
# retires specific versions for new API keys (gemini-2.5-flash now 404s for
# keys created recently), and the alias keeps working when that happens.
#
# The "lite" variant is chosen deliberately. The full flash model runs an
# internal reasoning pass first - measured here at 513 thinking tokens to
# produce 57 tokens of output - which burns free-tier quota and can return an
# empty response if the token budget is spent before any text is written. Lite
# does no thinking pass, costs about a fifth as many tokens, and rephrasing
# pre-computed facts needs no reasoning anyway.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

# Anthropic is supported as an alternative if ANTHROPIC_API_KEY is set instead.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")

LLM_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Turning SHAP output into stated facts
# ---------------------------------------------------------------------------
def _sentence_case(text: str) -> str:
    """Capitalise the first letter only - str.capitalize() lowercases the rest,
    which mangles descriptions like 'BMI category' into 'Bmi category'."""
    return text[:1].upper() + text[1:] if text else text


def risk_band(probability: float, threshold: float | None = None) -> str:
    """
    Describe the risk level *relative to the screening threshold*.

    Fixed cutoffs cannot do this job. The tuned diabetes threshold is 0.148, so
    a patient at 20% is flagged for follow-up - and a report calling that same
    patient "lower risk" in the next sentence contradicts the decision the
    system just made. Anchoring the bands to the threshold keeps the words and
    the flag telling the same story.

    Above the threshold, the flagged range is split in half: the lower half is
    "moderate", the upper half "high".
    """
    if threshold is None:
        threshold = 0.5  # no tuned threshold available; fall back to the default

    if probability < threshold:
        return "lower"
    if probability >= threshold + (1.0 - threshold) / 2:
        return "high"
    return "moderate"


def summarise_explanation(explanation: dict, top_n: int = 4) -> dict:
    """
    Reduce a SHAP explanation to the facts a report is allowed to state.

    Everything downstream - template or LLM - works from this dictionary and
    nothing else, so both narrators are grounded in identical information.
    """
    disease = explanation["disease"]

    def describe(items):
        return [
            {
                "factor": config.describe_feature(item["feature"]),
                "value": item["value"],
                "strength": abs(item["shap"]),
            }
            for item in items[:top_n]
        ]

    return {
        "disease": config.DISEASE_LABEL[disease],
        "probability": explanation["probability"],
        "band": risk_band(explanation["probability"],
                          explanation.get("threshold")),
        "raising": describe(explanation["increasing"]),
        "lowering": describe(explanation["decreasing"]),
    }


# ---------------------------------------------------------------------------
# Template narrator - always available
# ---------------------------------------------------------------------------
class TemplateNarrator:
    """Deterministic report writer. No network, no dependencies, no surprises."""

    name = "template"

    def narrate(self, summary: dict) -> str:
        disease = summary["disease"].lower()
        probability = summary["probability"]
        lines = [
            f"{summary['disease']} risk assessment",
            "",
            f"This patient's estimated {disease} risk is {probability:.0%}, "
            f"which the system classes as {summary['band']} risk.",
        ]

        if summary["raising"]:
            lines.append("")
            lines.append("What pushed this estimate up, in order of influence:")
            for index, item in enumerate(summary["raising"], start=1):
                lines.append(
                    f"  {index}. {_sentence_case(item['factor'])} "
                    f"(recorded value: {item['value']:g})"
                )

        if summary["lowering"]:
            lines.append("")
            # Deliberately not phrased as "in the patient's favour". These
            # factors pulled the score down relative to the model's *average*
            # patient - which, because training data was rebalanced with SMOTE,
            # is a far higher-risk person than the typical real one. Calling a
            # low income "protective" would be an unsupported medical claim.
            lines.append(
                "What pulled it back down, measured against the model's "
                "average patient:"
            )
            for index, item in enumerate(summary["lowering"], start=1):
                lines.append(
                    f"  {index}. {_sentence_case(item['factor'])} "
                    f"(recorded value: {item['value']:g})"
                )

        lines.append("")
        lines.append(
            "This is a statistical screening estimate based on population data, "
            "not a diagnosis. It is intended to help a clinician decide who to "
            "look at more closely."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM narrators - better prose, need an API key
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You turn the output of a medical risk-screening model into a short, clear \
explanation for a non-technical reader.

Rules you must follow:
- Use only the facts given to you. Never introduce a risk factor, number, \
diagnosis, or treatment that is not in the input.
- Never give medical advice or recommend treatment. You are explaining what a \
statistical model produced, not advising a patient.
- Do not describe the contributing factors as causes. They are correlations \
the model learned from population data.
- Factors listed as decreasing the estimate are measured against the model's \
average patient. Do not call them protective, healthy, or good news.
- Always say plainly that this is a screening estimate and not a diagnosis.
- Write 3 to 5 short sentences in plain English. No bullet points, no headings, \
no jargon, no markdown.\
"""


class _LLMNarrator:
    """Shared logic for any LLM-backed narrator: format the facts identically."""

    @staticmethod
    def _format_facts(summary: dict) -> str:
        def render(items):
            if not items:
                return "  (none)"
            return "\n".join(
                f"  - {item['factor']} (patient's value: {item['value']:g})"
                for item in items
            )

        return textwrap.dedent(f"""\
            Disease being screened for: {summary['disease']}
            Model's estimated risk: {summary['probability']:.0%} ({summary['band']} risk)

            Factors increasing this patient's estimated risk, most influential first:
            {render(summary['raising'])}

            Factors decreasing this patient's estimated risk, most influential first:
            {render(summary['lowering'])}
            """)

class GeminiNarrator(_LLMNarrator):
    """
    Sends the computed facts to Google Gemini and asks for fluent prose.

    Gemini is the default because its free tier requires no billing setup, so
    anyone cloning this repository can run the full pipeline at no cost.
    """

    name = "gemini"

    def __init__(self, model: str = GEMINI_MODEL, api_key: str | None = None):
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise RuntimeError(
                "The 'google-genai' package is required for Gemini reports. "
                "Install it with:  pip install google-genai"
            ) from error

        # The SDK also reads these automatically, but checking here produces a
        # clear message instead of a confusing failure at call time.
        key = (api_key
               or os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY"))
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set - cannot use the Gemini narrator."
            )

        self._types = types
        self.client = genai.Client(api_key=key)
        self.model = model

    def narrate(self, summary: dict) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=self._format_facts(summary),
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )

        text = (response.text or "").strip()
        if not text:
            # An empty response usually means a safety filter blocked it, or the
            # output-token budget was consumed before any text was produced.
            raise RuntimeError(
                "Gemini returned no text (possibly filtered or truncated)."
            )
        return text

    def available_models(self) -> list:
        """List models this key can use - handy when a model name goes stale."""
        return [m.name for m in self.client.models.list()]


class ClaudeNarrator(_LLMNarrator):
    """Alternative narrator using Anthropic's API, if ANTHROPIC_API_KEY is set."""

    name = "claude"

    def __init__(self, model: str = CLAUDE_MODEL, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as error:
            raise RuntimeError(
                "The 'anthropic' package is required for Claude reports. "
                "Install it with:  pip install anthropic"
            ) from error

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set - cannot use the Claude narrator."
            )

        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def narrate(self, summary: dict) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._format_facts(summary)}],
        )

        # A safety classifier can decline a request; that arrives as a normal
        # 200 with an empty content list, so stop_reason must be checked before
        # reading content.
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined to generate this report.")

        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_narrator(prefer_llm: bool = True):
    """
    Return the best available narrator.

    Tries Gemini first (free tier), then Claude, then falls back to the
    template. The fallback means report generation is never a hard dependency
    on a network call or an API key - the pipeline runs for anyone.
    """
    if prefer_llm:
        for narrator_class in (GeminiNarrator, ClaudeNarrator):
            try:
                return narrator_class()
            except RuntimeError:
                continue
    return TemplateNarrator()


def generate_report(explanation: dict, prefer_llm: bool = True,
                    narrator=None) -> str:
    """Produce a plain-English report for one patient's SHAP explanation."""
    summary = summarise_explanation(explanation)
    narrator = narrator or get_narrator(prefer_llm)

    try:
        return narrator.narrate(summary)
    except Exception as error:  # network failure, rate limit, refusal
        if isinstance(narrator, TemplateNarrator):
            raise
        print(f"    [LLM unavailable: {type(error).__name__}] "
              "falling back to the template report.")
        return TemplateNarrator().narrate(summary)


def main() -> None:
    """Generate an example report for one patient per disease."""
    import explainability
    import feature_engineering
    import train_models

    print("=" * 70)
    print("PHASE 8b: PLAIN-ENGLISH RISK REPORTS")
    print("=" * 70)

    narrator = get_narrator()
    print(f"\nNarrator in use: {narrator.name}")
    if narrator.name == "template":
        print("  (set GEMINI_API_KEY for LLM-written reports - the free tier at")
        print("   aistudio.google.com needs no billing. The template output")
        print("   below is the guaranteed fallback and needs no network access.)")
    else:
        model = getattr(narrator, "model", "?")
        print(f"  model: {model}")

    datasets = feature_engineering.prepare_all(verbose=False)
    for disease in config.DISEASES:
        dataset = datasets[disease]
        model = train_models.load(disease, train_models.GRADIENT_BOOSTING)
        position = explainability.pick_high_risk_patient(model, dataset)
        explanation = explainability.explain_patient(model, dataset, position)

        # Carry the tuned threshold across so this demo bands risk exactly the
        # way the deployed app does, instead of falling back to a generic 0.5.
        import deploy
        explanation["threshold"] = deploy.load(disease)["threshold"]

        print("\n" + "-" * 70)
        print(generate_report(explanation, narrator=narrator))

    print("-" * 70)


if __name__ == "__main__":
    main()
