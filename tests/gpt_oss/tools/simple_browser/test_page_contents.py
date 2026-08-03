from gpt_oss.tools.simple_browser.page_contents import get_domain, process_html


def test_get_domain_scheme_prefixed_urls():
    assert get_domain("https://httpbin.org/get") == "httpbin.org"
    assert get_domain("http://example.com/x") == "example.com"


def test_get_domain_bare_domain():
    assert get_domain("example.com") == "example.com"
    assert get_domain("example.com/path?q=1") == "example.com"


def test_get_domain_bare_host_containing_http():
    # Regression: the scheme check used to be `"http" not in url`, which treated
    # any scheme-less host that merely contained the substring "http" as already
    # schemed, so `urlparse().netloc` returned "".
    assert get_domain("httpbin.org") == "httpbin.org"
    assert get_domain("http.dev") == "http.dev"
    assert get_domain("shttp.io") == "shttp.io"
    assert get_domain("sub.httpbin.org") == "sub.httpbin.org"


def test_get_domain_bare_url_with_scheme_separator_later_in_url():
    # A "://" can appear in a path or query of an otherwise scheme-less URL, so
    # searching the whole string for a separator also misreads these as schemed.
    assert get_domain("example.com/cb?next=wss://socket") == "example.com"
    assert get_domain("example.com/redirect?to=https://other.com") == "example.com"
    assert get_domain("httpbin.org/cb?next=wss://socket") == "httpbin.org"


def test_get_domain_protocol_relative_url():
    assert get_domain("//example.com/path") == "example.com"


def test_same_domain_link_not_marked_external_for_bare_http_host():
    # A page opened via a bare host containing "http" previously resolved to
    # cur_domain == "", so every same-site link was rendered as external (with a
    # spurious `†domain` suffix the model uses to judge on-site vs off-site) and
    # the title fallback was empty.
    html = (
        "<html><body><p>"
        '<a href="https://httpbin.org/a">A</a> '
        '<a href="https://other.com/b">B</a>'
        "</p></body></html>"
    )
    page = process_html(html=html, url="httpbin.org", title=None)
    assert page.title == "httpbin.org"
    assert "【0†A】" in page.text  # same-domain link: no domain suffix
    assert "【1†B†other.com】" in page.text  # cross-domain link keeps its suffix
