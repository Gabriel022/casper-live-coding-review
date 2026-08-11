"""Tests for the Hacker News Research Brief pipeline.

Three focused tests covering the most failure-prone layers:

1. Discovery/parsing of the front-page HTML (rank, title, absolute URL,
   comment count) using an inline HTML fixture.
2. Record validation correctly flagging an article whose source_id is unknown.
3. Deduplication skipping candidates seen on previous runs and persisting the
   set of processed source IDs to a local JSON file.
"""

from main import (
    ArticleContent,
    filter_external,
    filter_unseen,
    load_seen_ids,
    parse_frontpage,
    save_seen_ids,
    validate_articles,
)

# Inline fixture mirroring the real Hacker News front-page markup. It contains
# one external story and one internal "Ask HN" story (an item?id link).
HN_FIXTURE = """
<html><body><table class="itemlist">
  <tr class='athing' id='40123456'>
    <td align="right" valign="top" class="title"><span class="rank">1.</span></td>
    <td valign="top" class="votelinks">
      <center><a id='up_40123456' href='vote?id=40123456&how=up'>
        <div class='votearrow' title='upvote'></div></a></center>
    </td>
    <td class="title"><span class="titleline">
      <a href="https://example.com/great-article">A Great Article</a>
      <span class="sitebit comhead"> (<a href="from?site=example.com">
        <span class="sitestr">example.com</span></a>)</span>
    </span></td>
  </tr>
  <tr><td colspan="2"></td><td class="subtext"><span class="subline">
    <span class="score" id="score_40123456">256 points</span>
    by <a href="user?id=alice" class="hnuser">alice</a>
    <span class="age" title="2024-01-01T00:00:00">
      <a href="item?id=40123456">3 hours ago</a></span>
    | <a href="hide?id=40123456">hide</a>
    | <a href="item?id=40123456">142&nbsp;comments</a>
  </span></td></tr>

  <tr class='athing' id='40999999'>
    <td align="right" valign="top" class="title"><span class="rank">2.</span></td>
    <td valign="top" class="votelinks">
      <center><a id='up_40999999' href='vote?id=40999999&how=up'>
        <div class='votearrow' title='upvote'></div></a></center>
    </td>
    <td class="title"><span class="titleline">
      <a href="item?id=40999999">Ask HN: What are you working on?</a>
    </span></td>
  </tr>
  <tr><td colspan="2"></td><td class="subtext"><span class="subline">
    <span class="score" id="score_40999999">42 points</span>
    by <a href="user?id=bob" class="hnuser">bob</a>
    <span class="age" title="2024-01-01T00:00:00">
      <a href="item?id=40999999">1 hour ago</a></span>
    | <a href="item?id=40999999">7&nbsp;comments</a>
  </span></td></tr>
</table></body></html>
"""


def test_parse_frontpage_extracts_metadata():
    """Discovery layer extracts rank, title, absolute URL, and comment count."""
    articles = parse_frontpage(HN_FIXTURE)
    assert len(articles) == 2

    first = articles[0]
    assert first.rank == 1
    assert first.title == "A Great Article"
    # Relative-free, fully absolute external URL.
    assert first.url == "https://example.com/great-article"
    assert first.comments == 142
    assert first.source_id == "40123456"
    assert first.points == 256
    assert first.author == "alice"
    assert first.hn_url == "https://news.ycombinator.com/item?id=40123456"

    # The internal "Ask HN" link is resolved to an absolute HN item URL.
    second = articles[1]
    assert second.rank == 2
    assert second.url == "https://news.ycombinator.com/item?id=40999999"
    assert second.comments == 7

    # Candidate filtering keeps only the external story.
    external = filter_external(articles)
    assert len(external) == 1
    assert external[0].url == "https://example.com/great-article"


def test_validate_articles_flags_unknown_source_id():
    """Validation layer drops and warns about an unknown source_id."""
    known_source_ids = {"40123456", "40999999"}

    good = ArticleContent(
        source_id="40123456",
        title="A Great Article",
        url="https://example.com/great-article",
        text="Real extracted body text for the first article.",
    )
    suspect = ArticleContent(
        source_id="40999999",
        title="Second Article",
        url="https://example.org/second",
        text="Real extracted body text for the second article.",
    )
    # Corrupt the second record's source_id to an unknown value.
    suspect.source_id = "unknown"

    valid, warnings = validate_articles([good, suspect], known_source_ids)

    assert len(valid) == 1
    assert valid[0].source_id == "40123456"
    assert any("unknown" in warning.lower() for warning in warnings)


def test_dedup_skips_seen_candidates(tmp_path):
    """Deduplication drops previously seen candidates and round-trips IDs."""
    candidates = parse_frontpage(HN_FIXTURE)  # source_ids 40123456 and 40999999
    seen_path = tmp_path / "seen_articles.json"

    # A missing file behaves as "nothing seen yet": all candidates are fresh.
    assert load_seen_ids(str(seen_path)) == set()
    assert len(filter_unseen(candidates, set())) == 2

    # Persist one processed ID, then read it back and confirm it is skipped.
    save_seen_ids({"40123456"}, str(seen_path))
    seen = load_seen_ids(str(seen_path))
    assert seen == {"40123456"}

    fresh = filter_unseen(candidates, seen)
    assert [candidate.source_id for candidate in fresh] == ["40999999"]
