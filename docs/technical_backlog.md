# Technical backlog

This shared backlog tracks repository-wide technical work that does not yet
belong to a specific experiment or project. Update the status, priority, and
owner in place as the team triages each item.

## Inference concurrency

- [ ] **Preserve useful exception behavior with `asyncio.TaskGroup`.**
  - Status: Open
  - Priority: TBD
  - Owner: Unassigned
  - Context: A failed query now escapes `_run_queries()` as an `ExceptionGroup`,
    even when there is only one underlying exception. Direct callers can no longer
    catch the original exception type, and experiment status records
    `error_type=ExceptionGroup` with a generic message instead of the underlying
    failure.
- [ ] **Decide whether one task per query is acceptable.**
  - Status: Open
  - Priority: TBD
  - Owner: Unassigned
  - Context: The current semaphore limits active query execution, but the stage
    still creates `O(query_count)` task objects. The previous worker pool created
    only `O(query_concurrency_limit)` tasks.
- [ ] **Expand inference concurrency tests.**
  - Status: Open
  - Priority: TBD
  - Owner: Unassigned
  - Context: Add coverage for query failure propagation, sibling cancellation,
    multiple simultaneous failures, and large query collections. Current tests
    cover successful concurrency and output ordering only.

## Logging

- [ ] **Persist configuration and preflight failures.**
  - Status: Open
  - Priority: TBD
  - Owner: Unassigned
  - Context: Failures before the immutable run directory and `run.log` handler are
    created currently go only to stderr or the enclosing Screen transcript.
- [ ] **Support generated project package namespaces in first-party logging.**
  - Status: Open
  - Priority: TBD
  - Owner: Unassigned
  - Context: The fixed `_FIRST_PARTY_LOGGERS` list knows the current repository
    packages, so `INFO` and `DEBUG` events from a newly generated project package
    may be filtered out.

## Templates and tests

- [ ] **Add a permanent template-generation smoke test.**
  - Status: Open
  - Priority: TBD
  - Owner: Unassigned
  - Context: Generate both `templates/retrieval-project` and
    `templates/retrieval-experiment`, compose the generated baseline and treatment
    configs, and validate the generated notebook. This flow has only been verified
    manually so far.
- [ ] **Make the configured `awesome-dev-tools` test suite self-contained.**
  - Status: Open
  - Priority: TBD
  - Owner: Unassigned
  - Context: The `retrieval-core` pytest configuration includes those tests, but
    its development dependencies do not provide `pyperclip`, so full collection
    can fail.

## Repository hygiene

- [ ] **Decide how to handle generated local state.**
  - Status: Open
  - Priority: TBD
  - Owner: Unassigned
  - Context: Decide whether `.pytest-tmp/` and `.frontend-slides/` should be ignored.
- [ ] **Decide whether to publish the onboarding presentation.**
  - Status: Open
  - Priority: TBD
  - Owner: Unassigned
  - Context: If `docs/r4_onboarding.html` is a published documentation artifact,
    link it from the documentation; otherwise keep or remove it as local output.
- [ ] **Resolve the query-repetition lockfile change.**
  - Status: Open
  - Priority: TBD
  - Owner: Unassigned
  - Context: Validate and commit or discard the modified
    `projects/query-repetition/uv.lock`.
