# AGENTS.md

## Scope

These instructions apply repo-wide unless a deeper `AGENTS.md` overrides them.

Codex works directly in the local checked-out repository. These rules are for local inspection and editing, not chat copy-paste delivery.

## Source Of Truth

1. The local checked-out workspace is the active source of truth for inspection and editing.
2. If the user provides a latest commit name, hash, tag, branch tip, or commit URL, treat it as an optional baseline reference for confirming the intended snapshot.
3. Use the local workspace for navigation, inspection, editing, and verification.
4. Do not ask the user to paste code or provide raw file URLs when the needed files are already present locally.
5. If the local workspace appears out of sync with the user's stated baseline reference, say so plainly before risky edits.
6. If the baseline cannot be confirmed well enough for a risky change, pause rather than guess.

## Working Style

1. Assume the user may describe desired behavior, bugs, or outcomes rather than naming files or code locations.
2. Codex must determine which files and which sections are relevant.
3. Only inspect, restructure, or edit the smallest relevant set of files needed for the current task.
4. Do not perform unrelated cleanup, style normalization, naming rewrites, or marker-system conversions in unrelated files.
5. If an older unrelated file uses a different marker style or label style such as `SECTION` instead of `SEGMENT`, or `SEGMENT` instead of `SECTION`, leave it alone unless that file is relevant to the current task.

## Plan First For Risky Work

For complex, ambiguous, or high-risk tasks, Codex should plan before editing.

Use a brief plan-first pause when the task includes things like:

1. cross-project changes
2. schema or migration changes
3. dependency changes
4. build, packaging, toolchain, or solution configuration changes
5. one-time restructuring passes
6. unclear requirements with non-obvious consequences
7. changes that touch multiple languages or multiple runtimes
8. changes that may affect external integrations, data formats, or generated outputs

The plan should identify the likely files involved, the risky parts, and the intended verification approach before edits begin.

## Approval Before Certain Changes

Codex should ask before doing any of the following, unless the user's request clearly requires it:

1. adding, removing, or upgrading dependencies or packages
2. editing project, solution, package, environment, CI, or build configuration files
3. deleting files
4. renaming files or moving files between directories
5. performing a one-time restructuring pass to establish or repair the section system
6. changing database schemas, migrations, or stored data formats
7. editing lockfiles in a way that is broader than the requested task

If the need becomes clear only after inspection, explain why and pause for approval.

## Protected Instruction File

1. Codex must not modify `AGENTS.md` itself unless the user explicitly asks.
2. If the user asks for a revision to `AGENTS.md`, treat that as a focused documentation/config task and avoid unrelated code edits in the same turn unless also requested.

## No Extra Artifacts

1. Do not create helper docs, scratch files, temp notes, sidecar instruction files, downloadable archives, or other non-source artifacts unless the user explicitly asks.
2. Do not create workaround files just to explain planned edits when a normal response will do.
3. If a non-source artifact is truly required for the task, pause and confirm unless the user already requested it.
4. Do not create large generated outputs, frame dumps, rendered media, bundled assets, database exports, or sample data unless the user explicitly asks.

## Official Edit Units

Codex edits files directly, but for structured file types below, the official edit unit remains the full section, full region, or full file defined by these rules.

When Codex changes a structured file, it must preserve the section system and edit at the official unit level, not as arbitrary partial fragments inside that unit, unless the file type rules explicitly make the full file the only allowed unit.

## Marker Label Preservation

1. `SECTION` and `SEGMENT` are both valid labels for replacement units.
2. In a relevant file, preserve that file's existing label style unless the task explicitly requires a one-time restructuring pass.
3. Do not rename `SECTION` to `SEGMENT`, or `SEGMENT` to `SECTION`, just for consistency.
4. For new files, default to `SECTION` unless the repo already has an obvious established convention for that file type.

## Rule 3A: Real Region Directives For Languages That Support Them

Use real collapsible regions wherever the language supports them well.

### C#

Use:

```csharp
#region SECTION A — <short title>
...
#endregion // SECTION A — <short title>
```

If the relevant file already uses `SEGMENT`, preserve that existing label style instead of converting it.

### Visual Basic

Use:

```vb
#Region "SECTION A - Title"
...
#End Region ' SECTION A - Title
```

If the relevant file already uses `SEGMENT`, preserve that existing label style instead of converting it.

Rules:

1. Use the same letter and title at both the start and end.
2. These region blocks are the official edit units.
3. If a language does not support these directives, follow the non-region rules below.

## Rule 3B: No Nested Replacement Units

1. No replacement unit may contain another replacement unit.
2. No C# `#region` may contain another C# `#region`.
3. No Visual Basic `#Region` may contain another Visual Basic `#Region`.
4. No JavaScript or TypeScript `//#region` may contain another JavaScript or TypeScript `//#region`.
5. No marker-based `SECTION START / SECTION END` block may contain another marker-based replacement unit.
6. A single replacement unit may contain one complete code block or multiple consecutive complete code blocks, as long as the whole unit remains a clean edit unit.
7. Do not wrap an entire file, namespace, or class in a replacement unit if more replacement units will live inside it.
8. Namespace and class declarations should normally sit outside replacement units unless the file already has an established sibling-section structure inside one wrapper body.
9. If a relevant file is not yet structured into safe non-nested replacement units, the first edit to that file may be a one-time restructuring pass, but only with user approval unless the request clearly requires it.

## Rule 3C: Official Marker Formats For File Types Without Rule 3A Directives

General rules:

1. Use the exact marker format required by the file type.
2. The marker must be on its own line.
3. The official unit is the full marker-wrapped section or region, from the opening marker line through the matching closing marker line, or from one section header to the line before the next section header when the file type uses section-header-only markers.
4. Never treat a partial fragment from inside a marked section as a standalone edit unit.
5. Use the same letter and title at both the start and end when the format includes both opening and closing markers.
6. For new files, default to `SECTION` labels unless the repo already clearly uses `SEGMENT` for that file type.
7. In existing files, preserve the file's current label word and marker style.

### Python `.py`

Use section-header-only markers:

```python
# ============================================================
# SECTION A — Title
# ============================================================
...
```

Official edit unit:
- from the section header through the line before the next section header, or EOF

If the file already uses `SEGMENT`, preserve that label style.

### Hash-Comment Section Languages

These use the same section-header-only structure as Python, but with the file's native single-line comment prefix.

This category includes:

1. `.sh`
2. `.bash`
3. `.zsh`
4. `.ps1`
5. `.rb`
6. `.pl`
7. `.r`
8. `.jl`
9. `.dockerfile`

Example for shell:

```sh
# ============================================================
# SECTION A — Title
# ============================================================
...
```

Official edit unit:
- from the section header through the line before the next section header, or EOF

If the file already uses `SEGMENT`, preserve that label style.

### Slash-Comment Section Languages

Use section-header-only markers with `//`:

```text
// ============================================================
// SECTION A — Title
// ============================================================
...
```

This category includes:

1. `.js`
2. `.jsx`
3. `.ts`
4. `.tsx`
5. `.java`
6. `.kt`
7. `.kts`
8. `.go`
9. `.rs`
10. `.swift`
11. `.c`
12. `.h`
13. `.cpp`
14. `.hpp`
15. `.cc`
16. `.hh`
17. `.dart`
18. `.scala`
19. `.groovy`
20. `.php`

Official edit unit:
- from the section header through the line before the next section header, or EOF

Special rule for JavaScript / TypeScript family:
- `//#region ...` / `//#endregion ...` is also allowed if the file already uses it
- preserve the existing style in that file rather than converting it

If the file already uses `SEGMENT`, preserve that label style.

### CSS / SCSS / LESS

Use:

```css
/* SECTION A START - Title */
...
/* SECTION A END - Title */
```

Official edit unit:
- from the opening marker through the closing marker

If the file already uses `SEGMENT`, preserve that label style.

### `.cshtml`

Use:

```cshtml
@* SECTION A START - Title *@
...
@* SECTION A END - Title *@
```

Official edit unit:
- from the opening marker through the closing marker

If the file already uses `SEGMENT`, preserve that label style.

### `.html` / `.xml` / `.xaml` / `.svg` / `.vue`

Use:

```html
<!-- SECTION A START - Title -->
...
<!-- SECTION A END - Title -->
```

Official edit unit:
- from the opening marker through the closing marker

If the file already uses `SEGMENT`, preserve that label style.

### `.sql`

Use:

```sql
-- SECTION A START - Title
...
-- SECTION A END - Title
```

Official edit unit:
- from the opening marker through the closing marker

If the file already uses `SEGMENT`, preserve that label style.

### `.md`

Use:

```md
<!-- SECTION A START - Title -->
...
<!-- SECTION A END - Title -->
```

Official edit unit:
- from the opening marker through the closing marker

If the file already uses `SEGMENT`, preserve that label style.

### Full-File-Only Types

Do not use markers. The only allowed edit unit is the full file.

This category includes:

1. `.json`
2. `.jsonc`
3. `.toml`
4. `.yaml`
5. `.yml`
6. `.csproj`
7. `.vbproj`
8. `.fsproj`
9. `.props`
10. `.targets`
11. `.sln`
12. `.slnx`
13. `.lock`
14. `.editorconfig`
15. `.gitignore`
16. `.gitattributes`
17. `package.json`
18. `package-lock.json`
19. `pnpm-lock.yaml`
20. `yarn.lock`
21. `Cargo.toml`
22. `Cargo.lock`
23. `pyproject.toml`
24. `poetry.lock`
25. `Pipfile`
26. `Pipfile.lock`
27. `requirements.txt`
28. `Dockerfile` when it is a standard Dockerfile without section markers already established
29. `Makefile`
30. `.env.example`

## Rule 3C1: Unknown File Types Require A Declaration

If the current task involves a file type that does not already have explicit rules in this document, Codex must stop before making code changes.

Codex must not guess.

Codex must respond with this exact format:

`DECLARATION: <file type> file type does not have clear rules and as per Rule 3C1, I must pause here before continuing.`

After that declaration, explicit rules for that file type must be established before editing proceeds.

## Rule 3D: Marker Placement Boundaries

In files that use marker-based sections or comment-region sections instead of Rule 3A directives, markers must be placed only on lines that sit between whole code blocks for that file type.

General rules:

1. An opening marker or section header must be immediately above the first whole code block in the section.
2. A closing marker must be immediately below the last whole code block in the section.
3. A marker must never be placed inside a code block.
4. A marker must never split a code block into pieces.
5. A section must never begin halfway through a declaration, statement, element, rule block, or expression.
6. A section must never end halfway through a declaration, statement, element, rule block, or expression.

### Section-Header-Only Code Languages

For Python, shell, PowerShell, Ruby, Perl, R, Julia, JavaScript, TypeScript, Java, Kotlin, Go, Rust, Swift, C-family, Dart, Scala, Groovy, and PHP:

A section header may be placed only:

1. above or below one or more consecutive full top-level declarations
2. above or below one or more consecutive full top-level executable statements or blocks
3. inside one existing top-level wrapper body only if the file already uses sibling sections at that scope
4. above or below one or more consecutive full class or module member declarations only if the file already uses sibling sections at that scope

A section header must never be placed:

1. inside a function body
2. inside a method body
3. inside a loop
4. inside an `if / else` block
5. inside a `switch / match / case` block
6. inside a `try / catch / finally` block
7. inside an object literal, array literal, map literal, or collection literal
8. inside an expression
9. inside a single declaration or statement
10. across a wrapper boundary

Special rule for JavaScript / TypeScript / JSX / TSX:
- imports and exports may exist in sectioned files, but a section must include them as whole top-level declarations and must not split an import group or export declaration

### CSS / SCSS / LESS

A marker may be placed only:

1. above or below one or more full top-level rule blocks
2. above or below one or more full top-level at-rules
3. above or below one or more full nested conditional blocks when already inside one established top-level wrapper structure

A marker must never be placed:

1. inside a selector block
2. inside an at-rule block
3. inside a declaration list
4. inside a media, supports, layer, keyframes, or container block

### `.cshtml`

A marker may be placed only:

1. above or below a full top-level markup block
2. above or below a full form
3. above or below a full section
4. above or below a full script block
5. above or below a full Razor block

A marker must never be placed:

1. inside an HTML element
2. inside a Razor expression
3. inside a Razor code block

### `.html` / `.xml` / `.xaml` / `.svg` / `.vue`

A marker may be placed only:

1. above or below a full element
2. above or below a full top-level structural block

A marker must never be placed inside an element.

### `.sql`

A marker may be placed only:

1. above or below a full `CREATE VIEW` statement
2. above or below a full `CREATE PROCEDURE` statement
3. above or below a full `CREATE FUNCTION` statement
4. above or below a full `CREATE TRIGGER` statement
5. above or below a full standalone DML block
6. above or below a full migration step

A marker must never be placed:

1. inside one of those statements
2. inside a `BEGIN / END` block
3. inside a conditional block
4. inside a loop

### `.md`

A marker may be placed only:

1. above or below a full heading section
2. above or below a full fenced code block
3. above or below a full HTML block

A marker must never be placed:

1. inside a fenced code block
2. inside an HTML block
3. inside a table
4. inside a list item

## Rule 3E: Whole-Block Section Content

1. A section or region may contain one whole block or multiple consecutive whole blocks.
2. Every block inside the section or region must be included from its own beginning to its own end.
3. A section or region must not begin halfway through a block.
4. A section or region must not end halfway through a block.
5. A section or region must not rely on an outer wrapper that contains multiple replacement units, unless the file already has an established sibling-section structure inside that wrapper and the boundaries remain clean.
6. If a file currently depends on an outer wrapper that would force multiple replacement units to sit inside it in an unsafe way, the first edit to that file must be a one-time restructuring pass.

## Segment And Section Size

1. Keep replacement units small enough to replace easily.
2. Ideal section size is 150 lines.
3. Hard cap is 300 lines unless truly unavoidable.
4. A section may contain one whole code block or multiple whole code blocks, but should still stay within a practical replacement size.
5. If needed, split large sections into new sections at clean boundaries.
6. If a section is split, clearly report the previous section name and the updated section names.

## Full-Section Integrity

1. Never omit closing braces, parentheses, brackets, semicolons, or trailing lines at the end of a section or region.
2. Never stop early.
3. Never cut off the bottom of a section or region.
4. Before finalizing an edit, verify that each changed section or region is complete and balanced.

## Stable Names

1. Keep identifiers and section names stable.
2. Do not rename files, classes, methods, functions, variables, modules, or section headers unless necessary.
3. If a section must be renamed, split, merged, or moved, clearly report that and include all affected units in the change summary.

## New Files

When creating a new file from scratch:

1. If the language supports `#region` or an equivalent real region directive, add section regions immediately using the exact syntax defined in Rule 3A.
2. If the language uses section-header-only markers, add section headers immediately using the exact syntax defined in Rule 3C.
3. If the language uses opening and closing marker pairs, add the file type's official marker format immediately using the exact syntax defined in Rule 3C.
4. Place markers only according to Rules 3D and 3E.
5. If the file type does not already have explicit rules in this document, follow Rule 3C1 before proceeding.

## Repo-Agnostic Boundaries

1. Prefer editing source files under the repo's real application or library directories such as:
   1. `src/`
   2. `app/`
   3. `server/`
   4. `client/`
   5. `lib/`
   6. `packages/`
   7. language- or framework-specific source directories that clearly contain active code
2. Do not edit generated outputs such as:
   1. `bin/`
   2. `obj/`
   3. `dist/`
   4. `build/`
   5. `.next/`
   6. `.nuxt/`
   7. `.parcel-cache/`
   8. `.turbo/`
   9. `.cache/`
   10. `.pytest_cache/`
   11. `coverage/`
   12. `target/`
   13. `out/`
   14. virtualenv directories
   15. package manager caches
3. Do not edit vendored, generated, or third-party dependency directories unless the task specifically requires it.
4. Treat fixtures, sample data, snapshots, exported files, recordings, and generated media as data, not as the default place for code changes.
5. In mixed-marker repos, preserve the existing marker naming and style in each relevant file unless restructuring is required.

## Verification And Done

When work is finished, verify using the smallest relevant set of commands for the task.

### Verification rules

1. Verify the smallest relevant scope first.
2. If only one package, project, or app was affected, start with that scope.
3. If shared code was changed, verify the directly dependent scopes when relevant.
4. Prefer existing repo scripts and standard project commands over invented commands.
5. Do not install new global tools just for verification unless the user explicitly asks.
6. If verification was not run, or could not be completed, say so clearly.

### Common verification examples

Use only what is relevant and available in the repo:

#### Python

```bash
python -m compileall <relevant_paths>
pytest
python -m <package_or_module> --help
```

#### JavaScript / TypeScript

```bash
npm test
npm run build
npm run lint
pnpm test
pnpm build
pnpm lint
yarn test
yarn build
yarn lint
```

#### .NET

```powershell
dotnet build <relevant_project_or_solution>
dotnet test <relevant_project_or_solution>
dotnet run --project <relevant_project>
```

#### Go

```bash
go test ./...
go build ./...
```

#### Rust

```bash
cargo test
cargo build
cargo run
```

#### Java / Kotlin

```bash
./gradlew build
./gradlew test
./gradlew run
mvn test
mvn package
```

#### Swift

```bash
swift build
swift test
```

#### C / C++ / CMake

Use the repo's documented build and test commands.

### Runtime checks when relevant

If the task affects runtime behavior, UI behavior, startup wiring, app flow, file I/O, CLI behavior, browser behavior, or external integrations, run the smallest relevant runtime or smoke check when practical.

If manual testing is relevant, say so clearly.

### Final Report Requirements

In the final report after edits, Codex must state:

1. exact relative file paths changed
2. exact section or region names changed, or explicitly say `full file`
3. what verification was run
4. what verification was not run
5. whether any restructuring pass occurred
6. whether any section or region was split, moved, or renamed

## If A Rule Conflicts With A Task

If a requested change would require breaking these rules, pause and explain the conflict plainly before editing.
