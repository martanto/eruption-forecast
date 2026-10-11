---
target: General Setup
total_score: 24
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 1
target_identity: "file:D:\\Projects\\eruption-forecast\\app\\frontend\\src\\pages\\GeneralSetupPage.tsx"
target_fingerprint: "sha256:b9d3d3b30800c21f9a4b68df66860cb57a7a77cd1a2008f18fe2a33bf9853bbc"
target_path: "D:\\Projects\\eruption-forecast\\app\\frontend\\src\\pages\\GeneralSetupPage.tsx"
timestamp: 2026-10-10T03-57-23Z
slug: app-frontend-src-pages-generalsetuppage-tsx
---
Method: dual-agent (A: setup_design_review · B: setup_evidence_review)

Target: app/frontend/src/pages/GeneralSetupPage.tsx. Mode: Operate.

## Design health

Scores use 4=excellent, 0=worst.

| Heuristic | Score | Finding |
|---|---:|---|
| System status | 2 | Shared busy state lacks operation-specific feedback. |
| Real-world match | 3 | Scientific language fits; SDS folder selection requires manual path knowledge. |
| User control | 3 | Manual/reset controls work; no selection/cancel flow exists. |
| Consistency | 3 | Coherent forest/neutral styling; native folder affordance is missing. |
| Error prevention | 2 | Path expectations and scan date constraints are not explicit. |
| Recognition | 2 | Technical keys displace useful contextual help. |
| Efficiency | 2 | Locating the archive requires typing or leaving the UI. |
| Minimalism | 3 | Calm visual system with repetitive small metadata. |
| Error recovery | 2 | Error feedback is global rather than near affected controls. |
| Help/documentation | 2 | Directory guidance is missing. |
| Total | 24/40 | Scoped polish needed; preserve incumbent identity. |

## Evidence and strengths

Independent design and detector/browser assessments. CLI detector found 0 findings in the page file. Live overlay found 22: 10 undersized-ui-text, 8 tiny-text, 3 low-contrast and 1 overused-font. The Inter warning is a low-relevance false positive for this operational interface. Desktop hierarchy, restrained surfaces and selected navigation work well. Mobile at 390×844 has no horizontal overflow.

## Priority issues

1. P1: Add a labeled SDS folder selector returning an absolute backend-machine path; preserve manual input and cancellation.
2. P2: Give absolute-path placeholders and concrete directory guidance.
3. P2: Group directories, station identity and execution settings without changing available parameters.
4. P2: Replace repetitive internal metadata with readable help; improve local placeholder contrast and touch targets.
5. P2: Add operation-specific progress, picker error/cancel feedback and scan-date validation near controls.

## Cognitive load and persona observations

Twelve equally weighted fields create interpretation work. Expert users still need direct path entry. Keyboard users need a labeled selector and focus restoration. Mobile users need larger controls and compact grouping. The emotional valley is locating and typing the archive path; clear selection/save feedback should resolve it.

## Scope

Preserve forest palette, serif headings, application navigation, scientific defaults and stage behavior. The user explicitly requested critique followed by polish, with SDS selection and absolute paths as priorities. No additional scope question is needed.
