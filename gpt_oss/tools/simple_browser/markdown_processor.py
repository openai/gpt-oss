"""
Markdown content processor for the simple browser tool.
Optimized replacement for HTML processing when content is already in markdown format.
"""

import re
from typing import Dict, Tuple, List

from .page_contents import PageContents, _replace_special_chars, remove_unicode_smp, arxiv_to_ar5iv, get_domain


def _merge_whitespace(text: str) -> str:
    """Merge whitespace like the HTML processor does."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_markdown_images(content: str) -> str:
    """Process markdown images into readable text format.
    
    Converts ![alt text](url) and ![alt text](url "title") to [Image: alt text] format.
    """
    # Pattern for markdown images: ![alt](src) or ![alt](src "title")
    img_pattern = re.compile(r'!\[([^\]]*)\]\([^\)]+(?:\s+"[^"]*")?\)')
    
    image_counter = 0
    
    def replace_image(match):
        nonlocal image_counter
        alt_text = match.group(1).strip()
        if alt_text:
            replacement = f"[Image {image_counter}: {alt_text}]"
        else:
            replacement = f"[Image {image_counter}]"
        image_counter += 1
        return replacement
    
    return img_pattern.sub(replace_image, content)


def process_markdown(
    content: str,
    url: str,
    title: str | None = None,
    display_urls: bool = False,
) -> PageContents:
    """
    Process markdown content into PageContents format.
    
    This is a simplified, faster alternative to process_html() for markdown content.
    
    Args:
        content: Markdown content
        url: Source URL
        title: Page title (optional)
        display_urls: Whether to display URL at top
        
    Returns:
        PageContents object with numbered links
    """
    # Apply same preprocessing as HTML processor
    content = remove_unicode_smp(content)
    content = _replace_special_chars(content)
    
    # Process markdown images first to avoid conflicts with link regex
    content = process_markdown_images(content)
    
    # Extract and number markdown links
    urls_dict, processed_content = extract_markdown_links(content, url)
    
    # Determine final title
    if title:
        final_title = title
    elif url and (domain := get_domain(url)):
        final_title = domain
    else:
        final_title = ""
    
    # Add URL display if requested - match HTML format exactly
    top_parts = []
    if display_urls:
        if url:
            top_parts.append(f"\nURL: {url}\n")
    
    # Basic content cleanup
    processed_content = clean_markdown_content(processed_content)
    
    final_content = "".join(top_parts) + processed_content
    
    return PageContents(
        url=url,
        text=final_content,
        urls=urls_dict,
        title=final_title,
        snippets=None,  # No snippets for direct content
        error_message=None,
    )


def extract_markdown_links(content: str, base_url: str = "") -> Tuple[Dict[str, str], str]:
    """
    Extract and number markdown links, converting them to citation format.
    
    Handles multiple markdown link formats:
    - [text](url)
    - [text](url "title") 
    - <url>
    - Reference links: [text][ref] with [ref]: url
    
    Args:
        content: Markdown content
        base_url: Base URL for context (used for domain filtering)
    
    Returns:
        Tuple of (urls_dict, processed_content)
        - urls_dict: {link_id: url}
        - processed_content: Content with links replaced by 【id†text†domain】
    """
    
    # Step 1: Extract reference definitions and remove them
    content, reference_map = _extract_reference_definitions(content)
    
    # Step 2: Find all link patterns
    link_matches = _find_all_links(content, reference_map)
    
    # Step 3: Number links and build replacement mapping
    urls_dict = {}
    url_to_id = {}  # Deduplicate identical URLs
    replacements = []
    
    base_domain = get_domain(base_url) if base_url else ""
    
    for match in link_matches:
        url = match['url']
        text = match['text']
        start = match['start']
        end = match['end']
        
        # Skip invalid URLs
        if not _is_valid_url(url):
            continue
            
        # Handle fragment-only links (like HTML processor)
        if url.startswith("#"):
            replacements.append({
                'start': start,
                'end': end,
                'replacement': text  # Just text, no citation
            })
            continue

        # Clean text like HTML processor does (merge whitespace and replace †)
        text = _merge_whitespace(text)
        text = text.replace("†", "‡")  # Match HTML processor behavior

        # Skip links with no text (like HTML processor does)
        # Use regex to check for meaningful text (not just whitespace or symbols)
        if not re.sub(r"【\@([^】]+)】", "", text).strip():
            # Remove the empty link syntax but don't create citation
            replacements.append({
                'start': start,
                'end': end,
                'replacement': text  # Just the (empty) text, no link
            })
            continue
            
        # Apply ArXiv to Ar5iv conversion
        url = arxiv_to_ar5iv(url)

        # Get or assign link ID
        if url in url_to_id:
            link_id = url_to_id[url]
        else:
            link_id = str(len(urls_dict))
            urls_dict[link_id] = url
            url_to_id[url] = link_id

        # Create replacement text
        domain = get_domain(url)
        if domain == base_domain:
            replacement = f"【{link_id}†{text}】"
        else:
            replacement = f"【{link_id}†{text}†{domain}】"

        replacements.append({
            'start': start,
            'end': end,
            'replacement': replacement
        })
    
    # Step 4: Apply replacements (in reverse order to maintain positions)
    processed_content = _apply_replacements(content, replacements)
    
    return urls_dict, processed_content


def _extract_reference_definitions(content: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract reference-style link definitions and remove them from content.
    
    Patterns:
    [ref]: url
    [ref]: url "title"
    [ref]: <url>
    """
    reference_map = {}
    
    # Pattern for reference definitions (case insensitive)
    ref_pattern = re.compile(
        r'^[ ]{0,3}\[([^\]]+)\]:[ \t]*<?([^\s>]+)>?(?:[ \t]+["\']([^"\']*)["\'])?[ \t]*$',
        re.MULTILINE | re.IGNORECASE
    )
    
    def replace_ref(match):
        ref_id = match.group(1).lower().strip()
        url = match.group(2).strip()
        reference_map[ref_id] = url
        return ''  # Remove the definition line
    
    content = ref_pattern.sub(replace_ref, content)
    
    return content, reference_map


def _find_all_links(content: str, reference_map: Dict[str, str]) -> List[Dict]:
    """Find all markdown links in content."""
    matches = []
    
    # Pattern 1: Inline links [text](url) or [text](url "title")
    inline_pattern = re.compile(
        r'\[([^\]]*)\]\(([^\s\)]+)(?:\s+"([^"]*)")?\)',
        re.IGNORECASE
    )
    
    for match in inline_pattern.finditer(content):
        original_text = match.group(1)  # Don't fallback to URL yet
        url = match.group(2)
        matches.append({
            'text': original_text,  # Keep original text (could be empty)
            'url': url,
            'start': match.start(),
            'end': match.end()
        })
    
    # Pattern 2: Reference links [text][ref]
    ref_pattern = re.compile(r'\[([^\]]*)\]\[([^\]]+)\]', re.IGNORECASE)
    
    for match in ref_pattern.finditer(content):
        ref_id = match.group(2).lower().strip()
        if ref_id in reference_map:
            matches.append({
                'text': match.group(1) or ref_id,  # Use ref as text if text is empty
                'url': reference_map[ref_id],
                'start': match.start(),
                'end': match.end()
            })
    
    # Pattern 3: Shortcut reference links [ref][]
    shortcut_pattern = re.compile(r'\[([^\]]+)\]\[\]', re.IGNORECASE)
    
    for match in shortcut_pattern.finditer(content):
        ref_id = match.group(1).lower().strip()
        if ref_id in reference_map:
            matches.append({
                'text': match.group(1),
                'url': reference_map[ref_id],
                'start': match.start(),
                'end': match.end()
            })
    
    # Pattern 4: Autolinks <url>
    autolink_pattern = re.compile(r'<(https?://[^\s>]+)>', re.IGNORECASE)
    
    for match in autolink_pattern.finditer(content):
        url = match.group(1)
        matches.append({
            'text': url,
            'url': url,
            'start': match.start(),
            'end': match.end()
        })
    
    # Sort by start position
    matches.sort(key=lambda x: x['start'])
    
    return matches


def _is_valid_url(url: str) -> bool:
    """Check if URL is valid and should be processed."""
    if not url:
        return False
    
    # Skip unwanted schemes
    if url.startswith(('mailto:', 'javascript:', 'tel:', 'ftp:')):
        return False
    
    # Must be HTTP/HTTPS or protocol-relative (case insensitive)
    url_lower = url.lower()
    if not (url_lower.startswith(('http://', 'https://', '//'))):
        return False
    
    return True


def _apply_replacements(content: str, replacements: List[Dict]) -> str:
    """Apply replacements in reverse order to maintain positions."""
    # Sort in reverse order by start position
    replacements.sort(key=lambda x: x['start'], reverse=True)
    
    for replacement in replacements:
        start = replacement['start']
        end = replacement['end']
        new_text = replacement['replacement']
        content = content[:start] + new_text + content[end:]
    
    return content


def clean_markdown_content(content: str) -> str:
    """Clean up markdown content after processing."""
    # Remove empty lines
    content = re.sub(r'^\s*$', '', content, flags=re.MULTILINE)
    
    # Normalize multiple newlines
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Clean up trailing whitespace
    content = content.strip()
    
    return content
