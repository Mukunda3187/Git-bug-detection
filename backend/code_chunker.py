"""
Code parsing and chunking stage.

Complete architecture:

GitHub Repository
        ↓
Source File Scanner
        ↓
Code Parser / Chunker
        ↓
Bug Detection
        ↓
Semantic Embeddings
        ↓
FAISS Vector Database
        ↓
Historical Retrieval
        ↓
LLM Analysis
        ↓
Fix Generation
        ↓
Fix Validation

The chunker converts source files into meaningful units such as:

    - functions
    - classes
    - methods
    - module-level code
    - generic code blocks

Python uses AST whenever possible.
Other languages use structure-aware brace parsing.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CHUNK_LINES = 80
MAX_CHUNK_LINES = 120

# Number of context lines added around a detected structural unit.
CONTEXT_LINES = 3


# ============================================================
# LANGUAGE DETECTION
# ============================================================

EXTENSION_LANGUAGE = {
    ".py": "Python",
    ".pyi": "Python",

    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",

    ".ts": "TypeScript",
    ".tsx": "TypeScript",

    ".java": "Java",

    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".hpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",

    ".cs": "C#",

    ".go": "Go",

    ".php": "PHP",

    ".rb": "Ruby",

    ".rs": "Rust",

    ".kt": "Kotlin",
    ".kts": "Kotlin",

    ".swift": "Swift",

    ".scala": "Scala",

    ".sql": "SQL",

    ".sh": "Shell",
    ".bash": "Shell",

    ".dart": "Dart",

    ".r": "R",

    ".lua": "Lua",

    ".m": "Objective-C",

    ".mm": "Objective-C++",

    ".groovy": "Groovy",

    ".ex": "Elixir",
    ".exs": "Elixir",

    ".hs": "Haskell",

    ".jl": "Julia",
}


def get_language(file_path: str) -> str:
    """
    Return the programming language based on file extension.
    """

    extension = os.path.splitext(file_path)[1].lower()

    return EXTENSION_LANGUAGE.get(
        extension,
        "Unknown",
    )


# ============================================================
# CHUNK CREATION
# ============================================================

def _make_chunk(
    *,
    file_path: str,
    language: str,
    chunk_type: str,
    name: str,
    start_line: int,
    end_line: int,
    code: str,
    parent: Optional[str] = None,
) -> Dict:

    return {
        "file": file_path,
        "language": language,
        "chunk_type": chunk_type,
        "name": name,
        "parent": parent,

        "line_start": max(1, int(start_line)),
        "line_end": max(
            int(start_line),
            int(end_line),
        ),

        "code": code.strip(),
    }


# ============================================================
# LINE HELPERS
# ============================================================

def _get_lines(source: str) -> List[str]:
    return source.splitlines()


def _safe_code(
    lines: List[str],
    start_line: int,
    end_line: int,
) -> str:

    if not lines:
        return ""

    start_index = max(
        0,
        start_line - 1,
    )

    end_index = min(
        len(lines),
        end_line,
    )

    if start_index >= end_index:
        return ""

    return "\n".join(
        lines[start_index:end_index]
    )


def _expand_context(
    start_line: int,
    end_line: int,
    total_lines: int,
) -> tuple[int, int]:

    start_line = max(
        1,
        start_line - CONTEXT_LINES,
    )

    end_line = min(
        total_lines,
        end_line + CONTEXT_LINES,
    )

    return start_line, end_line


def _limit_chunk(
    start_line: int,
    end_line: int,
) -> int:

    if end_line - start_line + 1 <= MAX_CHUNK_LINES:
        return end_line

    return (
        start_line
        + MAX_CHUNK_LINES
        - 1
    )


# ============================================================
# PYTHON AST CHUNKING
# ============================================================

def _python_node_name(node: ast.AST) -> str:

    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ),
    ):
        return getattr(
            node,
            "name",
            "<anonymous>",
        )

    return "<module>"


def _python_node_type(node: ast.AST) -> str:

    if isinstance(node, ast.ClassDef):
        return "class"

    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    ):
        return "function"

    return "module_code"


def _chunk_python(
    file_path: str,
    source: str,
) -> List[Dict]:

    lines = _get_lines(source)

    if not lines:
        return []

    try:
        tree = ast.parse(
            source,
            filename=file_path,
        )

    except SyntaxError as exc:

        # A syntax-error file is still useful to the
        # bug detector and LLM. Keep the whole file as
        # one chunk.

        return [
            _make_chunk(
                file_path=file_path,
                language="Python",
                chunk_type="file",
                name=os.path.basename(file_path),
                start_line=1,
                end_line=len(lines),
                code=source,
            )
        ]

    chunks: List[Dict] = []

    # --------------------------------------------------------
    # Recursive AST traversal
    # --------------------------------------------------------

    def visit(
        node: ast.AST,
        parent_name: Optional[str] = None,
    ):

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):

            start_line = getattr(
                node,
                "lineno",
                None,
            )

            end_line = getattr(
                node,
                "end_lineno",
                None,
            )

            if start_line is not None:

                if end_line is None:
                    end_line = start_line

                # Keep structural chunk bounded.
                limited_end = _limit_chunk(
                    start_line,
                    end_line,
                )

                # Add a small amount of context.
                context_start, context_end = (
                    _expand_context(
                        start_line,
                        limited_end,
                        len(lines),
                    )
                )

                code = _safe_code(
                    lines,
                    context_start,
                    context_end,
                )

                name = _python_node_name(
                    node
                )

                chunk_type = _python_node_type(
                    node
                )

                chunks.append(
                    _make_chunk(
                        file_path=file_path,
                        language="Python",
                        chunk_type=chunk_type,
                        name=name,
                        start_line=context_start,
                        end_line=context_end,
                        code=code,
                        parent=parent_name,
                    )
                )

                # Continue through nested definitions.
                child_parent = name

            else:
                child_parent = parent_name

        else:
            child_parent = parent_name

        for child in ast.iter_child_nodes(node):
            visit(
                child,
                child_parent,
            )

    # Traverse the complete tree.
    visit(tree)

    # --------------------------------------------------------
    # Top-level statements
    # --------------------------------------------------------

    structural_nodes = []

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            structural_nodes.append(node)

    # Collect module-level statements that are not
    # functions/classes.

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue

        start_line = getattr(
            node,
            "lineno",
            None,
        )

        end_line = getattr(
            node,
            "end_lineno",
            None,
        )

        if (
            start_line is None
            or end_line is None
        ):
            continue

        limited_end = _limit_chunk(
            start_line,
            end_line,
        )

        context_start, context_end = (
            _expand_context(
                start_line,
                limited_end,
                len(lines),
            )
        )

        code = _safe_code(
            lines,
            context_start,
            context_end,
        )

        if code.strip():

            chunks.append(
                _make_chunk(
                    file_path=file_path,
                    language="Python",
                    chunk_type="module_code",
                    name="<module>",
                    start_line=context_start,
                    end_line=context_end,
                    code=code,
                )
            )

    # --------------------------------------------------------
    # If AST produced nothing
    # --------------------------------------------------------

    if not chunks and source.strip():

        chunks.extend(
            _fallback_line_chunks(
                file_path,
                source,
                "Python",
            )
        )

    # --------------------------------------------------------
    # Remove exact duplicate ranges
    # --------------------------------------------------------

    chunks = _deduplicate_chunks(
        chunks
    )

    chunks.sort(
        key=lambda item: (
            item["line_start"],
            item["line_end"],
            item["chunk_type"],
        )
    )

    return chunks


# ============================================================
# GENERIC LANGUAGE STRUCTURE
# ============================================================

CLASS_PATTERNS = [

    re.compile(
        r"\bclass\s+"
        r"([A-Za-z_$][\w$]*)"
    ),

    re.compile(
        r"\binterface\s+"
        r"([A-Za-z_$][\w$]*)"
    ),

    re.compile(
        r"\bstruct\s+"
        r"([A-Za-z_$][\w$]*)"
    ),
]


FUNCTION_PATTERNS = [

    # JavaScript / TypeScript
    re.compile(
        r"\bfunction\s+"
        r"([A-Za-z_$][\w$]*)"
        r"\s*\("
    ),

    # Arrow functions
    re.compile(
        r"\b(?:const|let|var)\s+"
        r"([A-Za-z_$][\w$]*)"
        r"\s*=\s*(?:async\s*)?"
        r"\([^)]*\)\s*=>"
    ),

    # Python fallback
    re.compile(
        r"^\s*(?:async\s+)?"
        r"def\s+"
        r"([A-Za-z_]\w*)"
        r"\s*\("
    ),

    # Java / C / C++ / C# / Go / PHP
    re.compile(
        r"^\s*"
        r"(?:(?:public|private|protected|"
        r"static|async|virtual|override|"
        r"inline|extern|final|"
        r"abstract|synchronized)\s+)*"
        r"(?:[\w:<>,\[\].?]+\s+)+"
        r"([A-Za-z_]\w*)"
        r"\s*\([^;]*\)"
        r"\s*(?:throws\s+[^{]+)?"
        r"\{"
    ),

    # PHP function
    re.compile(
        r"\bfunction\s+"
        r"([A-Za-z_]\w*)"
        r"\s*\("
    ),

    # Ruby
    re.compile(
        r"^\s*def\s+"
        r"([A-Za-z_]\w*[!?=]?)"
    ),
]


# ============================================================
# BRACE PARSING
# ============================================================

def _strip_strings_and_comments(
    line: str,
) -> str:

    # This is deliberately lightweight.
    # It prevents most braces inside strings from
    # corrupting structural matching.

    result = re.sub(
        r'"(?:\\.|[^"\\])*"',
        '""',
        line,
    )

    result = re.sub(
        r"'(?:\\.|[^'\\])*'",
        "''",
        result,
    )

    result = re.sub(
        r"`(?:\\.|[^`\\])*`",
        "``",
        result,
    )

    # Remove simple // comments.
    result = re.sub(
        r"//.*$",
        "",
        result,
    )

    return result


def _find_brace_end(
    lines: List[str],
    start_index: int,
) -> int:

    depth = 0
    started = False

    for index in range(
        start_index,
        len(lines),
    ):

        cleaned = _strip_strings_and_comments(
            lines[index]
        )

        for char in cleaned:

            if char == "{":

                depth += 1
                started = True

            elif char == "}":

                if started:
                    depth -= 1

                    if depth <= 0:
                        return index

    # No matching closing brace.
    # Keep a bounded chunk.
    return min(
        len(lines) - 1,
        start_index
        + DEFAULT_CHUNK_LINES
        - 1,
    )


# ============================================================
# GENERIC CHUNKING
# ============================================================

def _chunk_generic(
    file_path: str,
    source: str,
) -> List[Dict]:

    language = get_language(
        file_path
    )

    lines = _get_lines(source)

    if not lines:
        return []

    chunks: List[Dict] = []

    ranges = []

    # --------------------------------------------------------
    # Classes / interfaces / structs
    # --------------------------------------------------------

    for index, line in enumerate(lines):

        match = None

        for pattern in CLASS_PATTERNS:

            candidate = pattern.search(
                line
            )

            if candidate:
                match = candidate
                break

        if not match:
            continue

        name = match.group(1)

        end = _find_brace_end(
            lines,
            index,
        )

        end = _limit_chunk(
            index + 1,
            end + 1,
        ) - 1

        start_line = index + 1
        end_line = end + 1

        context_start, context_end = (
            _expand_context(
                start_line,
                end_line,
                len(lines),
            )
        )

        code = _safe_code(
            lines,
            context_start,
            context_end,
        )

        chunks.append(
            _make_chunk(
                file_path=file_path,
                language=language,
                chunk_type="class",
                name=name,
                start_line=context_start,
                end_line=context_end,
                code=code,
            )
        )

        ranges.append(
            (
                index,
                end,
            )
        )

    # --------------------------------------------------------
    # Functions / methods
    # --------------------------------------------------------

    for index, line in enumerate(lines):

        match = None

        for pattern in FUNCTION_PATTERNS:

            candidate = pattern.search(
                line
            )

            if candidate:
                match = candidate
                break

        if not match:
            continue

        try:
            name = match.group(1)

        except (IndexError, AttributeError):
            name = "<function>"

        # Do not classify class declarations as functions.
        if re.search(
            r"\bclass\s+",
            line,
        ):
            continue

        # ----------------------------------------------------
        # Find function end.
        # ----------------------------------------------------

        if "{" in line:

            end = _find_brace_end(
                lines,
                index,
            )

        else:

            # Languages such as Ruby may terminate
            # with an explicit keyword rather than braces.
            end = _find_keyword_end(
                lines,
                index,
                language,
            )

        start_line = index + 1

        end_line = min(
            end + 1,
            start_line + MAX_CHUNK_LINES - 1,
        )

        context_start, context_end = (
            _expand_context(
                start_line,
                end_line,
                len(lines),
            )
        )

        code = _safe_code(
            lines,
            context_start,
            context_end,
        )

        chunks.append(
            _make_chunk(
                file_path=file_path,
                language=language,
                chunk_type="function",
                name=name,
                start_line=context_start,
                end_line=context_end,
                code=code,
            )
        )

        ranges.append(
            (
                index,
                end,
            )
        )

    # --------------------------------------------------------
    # No structural unit found.
    # --------------------------------------------------------

    if not chunks:

        chunks.extend(
            _fallback_line_chunks(
                file_path,
                source,
                language,
            )
        )

    # --------------------------------------------------------
    # Deduplicate and sort.
    # --------------------------------------------------------

    chunks = _deduplicate_chunks(
        chunks
    )

    chunks.sort(
        key=lambda item: (
            item["line_start"],
            item["line_end"],
            item["chunk_type"],
        )
    )

    return chunks


# ============================================================
# KEYWORD-BASED FUNCTION END
# ============================================================

def _find_keyword_end(
    lines: List[str],
    start_index: int,
    language: str,
) -> int:

    language_lower = language.lower()

    if language_lower == "ruby":

        for index in range(
            start_index + 1,
            len(lines),
        ):

            if re.match(
                r"^\s*end\s*$",
                lines[index],
            ):
                return index

    if language_lower == "elixir":

        for index in range(
            start_index + 1,
            len(lines),
        ):

            if re.match(
                r"^\s*end\s*$",
                lines[index],
            ):
                return index

    # Fallback.
    return min(
        len(lines) - 1,
        start_index
        + DEFAULT_CHUNK_LINES
        - 1,
    )


# ============================================================
# FALLBACK LINE CHUNKS
# ============================================================

def _fallback_line_chunks(
    file_path: str,
    source: str,
    language: str,
) -> List[Dict]:

    lines = _get_lines(source)

    chunks = []

    for start in range(
        0,
        len(lines),
        DEFAULT_CHUNK_LINES,
    ):

        end = min(
            len(lines),
            start + DEFAULT_CHUNK_LINES,
        )

        code = "\n".join(
            lines[start:end]
        )

        if not code.strip():
            continue

        chunks.append(
            _make_chunk(
                file_path=file_path,
                language=language,
                chunk_type="code_block",
                name=f"block_{start + 1}",
                start_line=start + 1,
                end_line=end,
                code=code,
            )
        )

    return chunks


# ============================================================
# DEDUPLICATION
# ============================================================

def _deduplicate_chunks(
    chunks: List[Dict],
) -> List[Dict]:

    seen = set()
    result = []

    for chunk in chunks:

        key = (
            chunk.get("file"),
            chunk.get("chunk_type"),
            chunk.get("name"),
            chunk.get("line_start"),
            chunk.get("line_end"),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(chunk)

    return result


# ============================================================
# PUBLIC API
# ============================================================

def chunk_source_file(
    file_path: str,
    source: str,
) -> List[Dict]:

    """
    Parse and chunk one source file.

    Returns:

        [
            {
                "file": "...",
                "language": "...",
                "chunk_type": "function",
                "name": "...",
                "parent": "...",
                "line_start": 10,
                "line_end": 40,
                "code": "..."
            }
        ]
    """

    if not source or not source.strip():
        return []

    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension in {
        ".py",
        ".pyi",
    }:

        return _chunk_python(
            file_path,
            source,
        )

    return _chunk_generic(
        file_path,
        source,
    )


def chunk_source_files(
    files,
) -> List[Dict]:

    """
    Chunk multiple source files.

    Input:

        [
            ("app.py", source_code),
            ("main.js", source_code)
        ]

    Output:

        Combined list of chunks.
    """

    all_chunks = []

    if not files:
        return all_chunks

    for item in files:

        try:

            if (
                not isinstance(item, tuple)
                and not isinstance(item, list)
            ):
                continue

            if len(item) < 2:
                continue

            file_path = item[0]
            source = item[1]

            chunks = chunk_source_file(
                file_path,
                source,
            )

            all_chunks.extend(
                chunks
            )

        except Exception as exc:

            print(
                f"[chunker] Failed to chunk "
                f"{item}: {exc}"
            )

    return all_chunks


# ============================================================
# FIND CHUNK FOR BUG
# ============================================================

def find_chunk_for_location(
    chunks: List[Dict],
    line_start: Optional[int],
    line_end: Optional[int] = None,
) -> Optional[Dict]:

    """
    Find the most relevant chunk for a detected bug.

    Used by the main pipeline to connect:

        detector finding
                ↓
        source-code chunk
                ↓
        RAG query
                ↓
        LLM analysis
    """

    if not chunks:
        return None

    try:
        start = int(line_start)

    except (
        TypeError,
        ValueError,
    ):
        return chunks[0]

    try:
        end = int(
            line_end
            if line_end is not None
            else start
        )

    except (
        TypeError,
        ValueError,
    ):
        end = start

    best_chunk = None
    best_overlap = -1

    for chunk in chunks:

        chunk_start = int(
            chunk.get(
                "line_start",
                0,
            )
        )

        chunk_end = int(
            chunk.get(
                "line_end",
                0,
            )
        )

        overlap_start = max(
            start,
            chunk_start,
        )

        overlap_end = min(
            end,
            chunk_end,
        )

        if overlap_start <= overlap_end:

            overlap = (
                overlap_end
                - overlap_start
                + 1
            )

            if overlap > best_overlap:

                best_overlap = overlap
                best_chunk = chunk

    # --------------------------------------------------------
    # No overlap: choose closest chunk.
    # --------------------------------------------------------

    if best_chunk is None:

        best_distance = None

        for chunk in chunks:

            chunk_start = int(
                chunk.get(
                    "line_start",
                    0,
                )
            )

            chunk_end = int(
                chunk.get(
                    "line_end",
                    0,
                )
            )

            if end < chunk_start:

                distance = (
                    chunk_start - end
                )

            elif start > chunk_end:

                distance = (
                    start - chunk_end
                )

            else:

                distance = 0

            if (
                best_distance is None
                or distance < best_distance
            ):

                best_distance = distance
                best_chunk = chunk

    return best_chunk


# ============================================================
# EMBEDDING / RAG CONTEXT
# ============================================================

def get_chunk_context(
    chunk: Dict,
) -> str:

    """
    Convert a chunk into a consistent text representation
    for semantic embedding and FAISS retrieval.
    """

    return (
        f"File: {chunk.get('file', '')}\n"
        f"Language: {chunk.get('language', '')}\n"
        f"Chunk type: {chunk.get('chunk_type', '')}\n"
        f"Name: {chunk.get('name', '')}\n"
        f"Parent: {chunk.get('parent') or ''}\n"
        f"Lines: "
        f"{chunk.get('line_start', '')}-"
        f"{chunk.get('line_end', '')}\n"
        f"Code:\n"
        f"{chunk.get('code', '')}"
    )


def get_chunk_id(
    chunk: Dict,
) -> str:

    """
    Stable identifier for connecting a chunk with
    embeddings and retrieved evidence.
    """

    return (
        f"{chunk.get('file', '')}:"
        f"{chunk.get('line_start', '')}-"
        f"{chunk.get('line_end', '')}:"
        f"{chunk.get('name', '')}"
    )