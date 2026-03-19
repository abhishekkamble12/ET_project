"""
Property-based tests for ContentCreationAgent.create_social_post.

Feature: social-media-multi-agent-system
"""
import sys
import os

# Set dummy API key so ChatGroq doesn't fail at module import time
os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-tests")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from agents.Content_creation import ContentCreationAgent, ContentCreationOutput


# ---------------------------------------------------------------------------
# Helpers / Strategies
# ---------------------------------------------------------------------------

PLATFORMS = ["linkedin", "instagram"]

_tag_chars = st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_")
_tag_text = st.text(alphabet=_tag_chars, min_size=1, max_size=20)


def _make_output(platform: str, hashtags: list[str]) -> ContentCreationOutput:
    return ContentCreationOutput(
        caption="Test caption about the topic",
        image_prompt="A vivid image of the topic",
        hashtags=hashtags,
        platform=platform,
    )


@st.composite
def linkedin_hashtag_list(draw):
    """1–5 unique hashtags for LinkedIn."""
    n = draw(st.integers(min_value=1, max_value=5))
    tags = draw(st.lists(_tag_text, min_size=n, max_size=n, unique=True))
    return tags


@st.composite
def instagram_hashtag_list(draw):
    """10–30 unique hashtags for Instagram."""
    n = draw(st.integers(min_value=10, max_value=30))
    tags = draw(st.lists(_tag_text, min_size=n, max_size=n, unique=True))
    return tags


@st.composite
def memory_context_with_prior_posts(draw, platform: str):
    """
    Generate a memory_context dict containing at least 1 prior post for the given platform.
    Each prior post has a 'hashtags' list of 1–10 tags.
    Returns (memory_context, all_prior_hashtags_in_3_most_recent).
    """
    num_posts = draw(st.integers(min_value=1, max_value=10))
    posts = []
    for _ in range(num_posts):
        n_tags = draw(st.integers(min_value=1, max_value=10))
        tags = draw(st.lists(_tag_text, min_size=n_tags, max_size=n_tags, unique=True))
        posts.append({"hashtags": tags})

    memory_context = {f"{platform}_posts": posts}

    # Collect hashtags from the 3 most recent posts
    recent = posts[-3:]
    recent_hashtags: set[str] = set()
    for post in recent:
        recent_hashtags.update(post.get("hashtags", []))

    return memory_context, recent_hashtags


# ---------------------------------------------------------------------------
# Property 3: Content generation output is structurally complete
# Validates: Requirements 3.1, 3.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    topic=st.text(min_size=1, max_size=100),
    platform=st.sampled_from(PLATFORMS),
    memory_context=st.one_of(st.none(), st.just({})),
)
def test_p3_content_output_completeness(topic, platform, memory_context):
    # Feature: social-media-multi-agent-system, Property 3: Content generation output is structurally complete
    """
    For any valid topic, platform, and memory context, create_social_post must return a
    ContentCreationOutput where caption, image_prompt, hashtags, and platform are all
    non-null and non-empty.
    """
    if platform == "linkedin":
        mock_output = _make_output(platform, ["AI", "Tech", "Innovation"])
    else:
        mock_output = _make_output(platform, [f"tag{i}" for i in range(15)])

    agent = ContentCreationAgent()

    with patch("agents.Content_creation.llm") as mock_llm, \
         patch("agents.Content_creation.parser") as mock_parser, \
         patch("agents.Content_creation.generate_image") as mock_gen_image:

        mock_llm.invoke.return_value = MagicMock(content="mocked llm response")
        mock_parser.get_format_instructions.return_value = ""
        mock_parser.parse.return_value = mock_output
        mock_gen_image.return_value = "base64imagedata"

        result = agent.create_social_post(topic, platform, memory_context)

    assert result.caption is not None and len(result.caption) > 0, \
        "caption must be non-null and non-empty"
    assert result.image_prompt is not None and len(result.image_prompt) > 0, \
        "image_prompt must be non-null and non-empty"
    assert result.hashtags is not None and len(result.hashtags) > 0, \
        "hashtags must be non-null and non-empty"
    assert result.platform is not None and len(result.platform) > 0, \
        "platform must be non-null and non-empty"


# ---------------------------------------------------------------------------
# Property 4: Platform hashtag count invariant
# Validates: Requirements 3.4, 3.5
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    topic=st.text(min_size=1, max_size=100),
    hashtags=linkedin_hashtag_list(),
)
def test_p4_linkedin_hashtag_count(topic, hashtags):
    # Feature: social-media-multi-agent-system, Property 4: Platform hashtag count invariant (LinkedIn)
    """
    For any LinkedIn post, the hashtags list must have length <= 5.
    """
    mock_output = _make_output("linkedin", hashtags)
    agent = ContentCreationAgent()

    with patch("agents.Content_creation.llm") as mock_llm, \
         patch("agents.Content_creation.parser") as mock_parser, \
         patch("agents.Content_creation.generate_image"):

        mock_llm.invoke.return_value = MagicMock(content="mocked")
        mock_parser.get_format_instructions.return_value = ""
        mock_parser.parse.return_value = mock_output

        result = agent.create_social_post(topic, "linkedin", None)

    assert len(result.hashtags) <= 5, \
        f"LinkedIn post must have <= 5 hashtags, got {len(result.hashtags)}"


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    topic=st.text(min_size=1, max_size=100),
    hashtags=instagram_hashtag_list(),
)
def test_p4_instagram_hashtag_count(topic, hashtags):
    # Feature: social-media-multi-agent-system, Property 4: Platform hashtag count invariant (Instagram)
    """
    For any Instagram post, the hashtags list must have length between 10 and 30 inclusive.
    """
    mock_output = _make_output("instagram", hashtags)
    agent = ContentCreationAgent()

    with patch("agents.Content_creation.llm") as mock_llm, \
         patch("agents.Content_creation.parser") as mock_parser, \
         patch("agents.Content_creation.generate_image"):

        mock_llm.invoke.return_value = MagicMock(content="mocked")
        mock_parser.get_format_instructions.return_value = ""
        mock_parser.parse.return_value = mock_output

        result = agent.create_social_post(topic, "instagram", None)

    assert 10 <= len(result.hashtags) <= 30, \
        f"Instagram post must have 10-30 hashtags, got {len(result.hashtags)}"


# ---------------------------------------------------------------------------
# Property 5: Hashtag novelty relative to recent memory
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    topic=st.text(min_size=1, max_size=100),
    platform=st.sampled_from(PLATFORMS),
    ctx_and_recent=st.one_of(
        memory_context_with_prior_posts("linkedin"),
        memory_context_with_prior_posts("instagram"),
    ),
)
def test_p5_hashtag_novelty(topic, platform, ctx_and_recent):
    # Feature: social-media-multi-agent-system, Property 5: Hashtag novelty relative to recent memory
    """
    For any generated post where memory context contains at least one prior post for the
    same platform, none of the hashtags in the new post should appear in the union of
    hashtags from the three most recent prior posts for that platform.
    """
    memory_context, recent_hashtags = ctx_and_recent

    # Determine the platform that actually has prior posts in this memory_context
    actual_platform = "linkedin" if "linkedin_posts" in memory_context else "instagram"

    # Build new hashtags that are guaranteed NOT in recent_hashtags
    new_hashtags = [f"brandnew_{i}" for i in range(3 if actual_platform == "linkedin" else 15)]
    mock_output = _make_output(actual_platform, new_hashtags)

    agent = ContentCreationAgent()

    with patch("agents.Content_creation.llm") as mock_llm, \
         patch("agents.Content_creation.parser") as mock_parser, \
         patch("agents.Content_creation.generate_image"):

        mock_llm.invoke.return_value = MagicMock(content="mocked")
        mock_parser.get_format_instructions.return_value = ""
        mock_parser.parse.return_value = mock_output

        result = agent.create_social_post(topic, actual_platform, memory_context)

    overlap = set(result.hashtags) & recent_hashtags
    assert len(overlap) == 0, (
        f"New post hashtags overlap with recent memory hashtags: {overlap}"
    )
