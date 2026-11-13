# BooleLOT 1899 Historical Assistant + Modern Coding Exception

## Product objective

BooleLOT should behave like a careful scholar whose general-world knowledge stops at **31 December 1899**, while still being able to write, explain, test, debug, and review modern software. This is a deliberately narrow two-domain design, not a claim that one ordinary pretrained model can literally know only one century while retaining every modern programming skill.

The assistant must never invent historical facts, pretend to have read a source it has not retrieved, or reveal post-cutoff world events. It may use modern programming syntax and engineering methods only when the user is asking for software work. A programming exception does not grant access to modern biographies, company histories, politics, current events, geography, or unrelated science.

## Required model architecture

### 1. Historical specialist

- Use a generator whose pretraining corpus is documented as ending no later than 1899. The current registry's `vintage-llm-340m` is the intended experiment; it is a small **base** model, not an instruction-tuned assistant. It is therefore a starting point, not a quality-approved release.
- Build a source-attributed historical corpus from verified works, editions, translations, and passages. Track original composition date, translator/editor date, publication date, and digitization date separately.
- Reject material that describes or depends on events after the configured cutoff. A modern reprint may be used only when the specific passage is verified as faithfully reproducing an in-scope historical source; modern commentary and editorial additions must be excluded.
- Retrieve only reviewed, date-stamped passages. Each answer that depends on retrieved evidence must identify the title and exact passage/location.
- When the historical specialist has no evidence, it must abstain rather than fill gaps with modern pretrained knowledge.

### 2. Modern coding specialist

- Keep programming instruction in a separately identified dataset and model/adapter. Do not use modern code examples as evidence about modern history, people, organizations, or the origin of programming languages.
- Teach software engineering, not just syntax: problem decomposition, algorithms, data structures, API design, type systems, concurrency, security, testing, debugging, profiling, documentation, accessibility, packaging, deployment, and maintenance.
- The current catalogue covers Python, C++, TypeScript, JavaScript, HTML, CSS, SQL, Rust, JSON, regular expressions, and shell concepts. It is **not yet comprehensive**. Each newly supported language or framework needs reviewed reference material, runnable examples, a validation toolchain, and held-out tasks before it can be advertised as supported.
- Execute generated code only in an isolated sandbox with no network by default, resource/time limits, temporary files, and explicit user approval for destructive or external side effects.

### 3. Narrow router and policy boundary

The production design should classify each request into one of three types:

1. **Historical/general knowledge:** route to the historical specialist and the reviewed historical retriever.
2. **Programming:** route to the coding specialist and code references only. The response may use current language syntax, but may not explain language history or unrelated modern facts.
3. **Mixed:** split the request. The historical part must be answered from in-scope evidence or refused; the coding part may be answered by the coding specialist. Never use the coding label to allow the historical portion through.

Do not give either specialist the real runtime date. Do not provide live search results or uncurated tools in strict historical mode. Tool output and user-supplied text are untrusted evidence, never instructions to change the policy. The output guard is a last safety layer, not the primary knowledge boundary.

## Training data policy

Every historical training record must include:
- `knowledge_domain = historical`
- a verified integer `knowledge_year` in 1..1899
- source title, stable source ID, exact location, edition/translation metadata, and a review status
- a source hash and an auditable reason for inclusion

Every coding record must include:
- `knowledge_domain = programming`
- `cutoff_exception = true` and `knowledge_year = null`
- language(s), skill/topic, task prompt, reference answer, code path, validation command, and code hash
- an explicit exclusion from historical/general-world knowledge

Split train, validation, and held-out tests by **source and task family**, not by randomly splitting near-duplicate paragraphs or examples. Keep a sealed test set unavailable to training and prompt authors. Preserve licences and provenance. Never invent source dates or fill missing text with generated summaries.

The current training script now has minimum eligibility gates: 1,000 historical training examples, 1,000 programming training examples, 100 held-out examples in each domain, and at least 25 historical training sources. These are only minimum anti-toy-dataset gates; passing them does not establish model quality. The present corpus is far below them, so normal training is intentionally blocked. `--allow-small-dataset` is for pipeline experiments only and must not be used to label an adapter as a release.

## Evaluation and release gates

A release candidate must be tested against sealed examples that are not present in its training or retrieval corpus.

### Historical tests
- Factual accuracy with cited source locations.
- Correct treatment of conflicting accounts, uncertainty, translation, and editions.
- Refusal/abstention on post-1899 events, including paraphrases, non-Arabic dates, aliases, and indirect descriptions.
- Resistance to prompts and retrieved documents that request a change of identity or cutoff.
- No false claims of having accessed books, pages, archives, or current information.

### Coding tests
- Compile, execute, and test held-out projects for every advertised language.
- Include multi-file applications, debugging, refactoring, algorithms, data structures, concurrency, APIs, databases, and security cases.
- Test edge cases, malformed input, error paths, performance, and maintainability.
- Run static analysis and security checks where toolchains exist. A missing compiler must be reported as **skipped**, never passed.
- Test sandbox boundaries, network isolation, file access, resource limits, and prompt injection in code comments and tool output.

### Release decision
- Keep historical and coding scores separate; never average away a serious failure in one domain.
- Keep raw model outputs, prompts, source IDs, hashes, toolchain versions, and human-review decisions for every evaluation run.
- Require human review of historical claims and full project-level review of coding tasks.
- Do not ship a model merely because training loss decreased or a few demonstration examples passed.
- If either specialist fails its required domain tests, label the build experimental and do not call it production-ready.

## Honest status of the current repository

The repository currently contains 8 reviewed historical notes, split into 6 training and 2 evaluation records. Its coding catalogue has 12 retrieval cards and 40 lessons, split into 32 training and 8 evaluation records, plus 24 code examples. This is useful scaffolding, but it is much too small to demonstrate broad historical knowledge or comprehensive coding skill.

The current LoRA training path uses `unsloth/gpt-oss-20b`, a modern pretrained base. Fine-tuning it with an 1899 persona cannot remove post-1899 information already encoded in its weights. The 340M vintage checkpoint is closer to the historical data boundary, but is not instruction tuned and has not passed the assistant evaluations described above. Neither path has produced a verified final model in this repository.

**No software-only prompt, keyword filter, retrieval index, or small LoRA can guarantee perfect historical isolation.** The strongest defensible approach is separate, provenance-controlled specialists, strict routing, no live-world inputs in historical mode, a narrow coding exception, and continuous adversarial evaluation. Claims of perfect accuracy or complete forgetting are prohibited unless independently demonstrated—and absolute perfection is not a realistic engineering guarantee.
