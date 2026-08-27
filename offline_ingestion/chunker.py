"""把结构化 Markdown 切成适合检索、并且可以精确引用的文本块。

整个模块分成两层，初学者可以按下面的顺序理解：

1. :func:`read_markdown_blocks` 只负责理解 Markdown 结构。它识别文档标题、
   二到四级标题、法条标题和 PDF 页码标记，生成“父块”。
2. :func:`build_markdown_chunks` 再把过长父块拆成“子块”。拆分优先选择
   自然段和中文句末标点；只有一个句子本身已经过长时，才按字符兜底。

这样做有三个目的：不把相邻法条混在一起；每个子块都重复法规名称和
章/条路径；PDF 子块永远不跨真实页码。后两点直接决定检索结果能否被
用户核对，所以这里没有根据 Word 排版或网页打印效果猜测页码。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from shared.models import SourceDocument, TextChunk


# 转换器约定：一级标题是文档名；二到四级标题是正文结构。
MARKDOWN_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
ARTICLE_TITLE = re.compile(r"^(第\s*[^\s]+\s*条)\s*$")
LEGAL_CHAPTER_TITLE = re.compile(r"^第\s*[^\s]+\s*章(?:\s|$)")
PAGE_MARKER = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$")

# 过长自然段优先在这些“句子结束位置”后拆分。逗号不是完整句末，只有
# 找不到句号、分号等位置时才用作次优边界。
STRONG_BOUNDARY = re.compile(r"(?<=[。！？；;])")
WEAK_BOUNDARY_CHARACTERS = "，,、：:"


@dataclass(frozen=True)
class MarkdownSegment:
    """父块中的一段正文，以及它来自哪一页 PDF。

    DOCX 和 HTML 没有稳定的原始页码，因此 ``page_number`` 合法地为
    ``None``。空值比伪造页码更适合法律引用。
    """

    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class MarkdownBlock:
    """一个不能与相邻结构混合的 Markdown 父块。

    ``structure_type`` 目前只有三种：

    - ``article``：第X条；
    - ``section``：普通通知、附件或目录中的标题段；
    - ``preamble``：第一个正文标题以前的发文说明等内容。
    """

    structure_type: str
    title: str
    section_path: tuple[str, ...]
    segments: tuple[MarkdownSegment, ...]
    heading_level: int = 0
    chapter: str = ""
    article_number: str = ""
    page_number: int | None = None

    @property
    def body(self) -> str:
        """返回完整正文，供 Markdown 审计和测试复用。"""

        return "\n".join(segment.text for segment in self.segments).strip()


@dataclass(frozen=True)
class MarkdownArticle:
    """兼容旧审计代码的法条视图。

    新解析器也支持普通章节，但 ``audit_markdown.py`` 仍只需要统计法条，
    所以保留这个小数据结构，避免让审计模块了解分块内部实现。
    """

    chapter: str
    article_number: str
    segments: tuple[MarkdownSegment, ...]
    page_number: int | None = None

    @property
    def body(self) -> str:
        return "\n".join(segment.text for segment in self.segments).strip()


def split_fixed_characters(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按固定字符窗口切分文本，作为兼容接口和最后一道兜底。

    新的 Markdown 主流程不会一开始就调用它；只有一个句子本身超过可用
    长度时才需要字符级切分。参数检查可以避免 ``step <= 0`` 导致死循环。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    cleaned = " ".join(text.split())
    step = chunk_size - overlap
    return [
        cleaned[start : start + chunk_size]
        for start in range(0, len(cleaned), step)
        if cleaned[start : start + chunk_size]
    ]


def _normalized_article_title(title: str) -> str | None:
    """识别并去掉“第 九 条”中的版式空格。"""

    match = ARTICLE_TITLE.match(title)
    return "".join(match.group(1).split()) if match else None


def read_markdown_blocks(markdown_path: Path) -> tuple[str, list[MarkdownBlock]]:
    """解析转换器生成的 Markdown，同时保留标题层级和真实 PDF 页码。

    这里故意不依赖大型 Markdown 解析库。项目生成的 Markdown 有一个很小
    且稳定的契约：YAML front matter、``#`` 标题和 ``<!-- page: N -->``。
    直接读取这个契约更容易测试，也更适合作为简历项目向面试官解释。
    """

    document_title = ""
    active_page: int | None = None
    in_front_matter = False
    front_matter_seen = False

    # ``heading_stack[2]`` 表示最近的二级标题。遇到新的二级标题时，旧的
    # 三级和四级标题会被清掉，防止路径串到下一个章节。
    heading_stack: dict[int, str] = {}
    active_type = ""
    active_title = ""
    active_level = 0
    active_chapter = ""
    active_article = ""
    active_path: tuple[str, ...] = ()
    active_start_page: int | None = None
    segments: list[MarkdownSegment] = []
    blocks: list[MarkdownBlock] = []

    def finish_block() -> None:
        """把当前可变状态固化为一个父块，然后清空正文缓冲区。"""

        nonlocal active_type, active_title, active_level, active_chapter
        nonlocal active_article, active_path, active_start_page, segments
        if active_type and any(segment.text.strip() for segment in segments):
            blocks.append(
                MarkdownBlock(
                    structure_type=active_type,
                    title=active_title,
                    section_path=active_path,
                    segments=tuple(segments),
                    heading_level=active_level,
                    chapter=active_chapter,
                    article_number=active_article,
                    page_number=active_start_page,
                )
            )
        active_type = ""
        active_title = ""
        active_level = 0
        active_chapter = ""
        active_article = ""
        active_path = ()
        active_start_page = None
        segments = []

    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines):
        stripped = raw_line.strip()

        # 只把文件开头的第一对 --- 当作 front matter。正文中的水平分隔线
        # 不应让解析器错误地重新进入 YAML 状态。
        if stripped == "---" and not front_matter_seen and line_number == 0:
            in_front_matter = True
            front_matter_seen = True
            continue
        if stripped == "---" and in_front_matter:
            in_front_matter = False
            continue
        if in_front_matter or not stripped:
            continue

        page_match = PAGE_MARKER.match(stripped)
        if page_match:
            active_page = int(page_match.group(1))
            continue

        heading_match = MARKDOWN_HEADING.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            if level == 1:
                # 一级标题是检索上下文，不是一个独立正文块。
                document_title = title
                continue

            finish_block()
            for stale_level in range(level, 5):
                heading_stack.pop(stale_level, None)
            heading_stack[level] = title

            article_number = _normalized_article_title(title)
            active_type = "article" if article_number else "section"
            active_title = article_number or title
            active_level = level
            active_article = article_number or ""
            active_chapter = next(
                (
                    heading_stack[parent_level]
                    for parent_level in range(level - 1, 1, -1)
                    if LEGAL_CHAPTER_TITLE.match(heading_stack.get(parent_level, ""))
                ),
                "",
            )
            active_path = tuple(
                heading_stack[path_level]
                for path_level in range(2, level + 1)
                if path_level in heading_stack
            )
            active_start_page = active_page
            continue

        if not active_type:
            # 标题后、首个二级标题前的发文号或说明也可能对回答有用，不能像
            # 旧版一样静默丢弃。它们统一归入一个 preamble 父块。
            active_type = "preamble"
            active_title = "正文前言"
            active_path = ("正文前言",)
            active_start_page = active_page
        segments.append(MarkdownSegment(stripped, active_page))

    finish_block()
    return document_title, blocks


def read_markdown_articles(markdown_path: Path) -> list[MarkdownArticle]:
    """返回 Markdown 中的法条，供现有审计代码保持向后兼容。"""

    _, blocks = read_markdown_blocks(markdown_path)
    return [
        MarkdownArticle(
            chapter=block.chapter,
            article_number=block.article_number,
            segments=block.segments,
            page_number=block.page_number,
        )
        for block in blocks
        if block.structure_type == "article"
    ]


def _group_segments_by_page(block: MarkdownBlock) -> list[tuple[int | None, list[str]]]:
    """按连续真实页码分组，保证后面的一个子块只对应一页 PDF。"""

    groups: list[tuple[int | None, list[str]]] = []
    for segment in block.segments:
        page_number = segment.page_number if segment.page_number is not None else block.page_number
        if groups and groups[-1][0] == page_number:
            groups[-1][1].append(segment.text)
        else:
            groups.append((page_number, [segment.text]))
    return groups


def _split_oversized_unit(text: str, max_chars: int) -> list[str]:
    """把单个超长段落尽量沿句末标点切开，最后才使用字符边界。"""

    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []

    # 先把自然段拆成完整句子。若句子仍然过长，再在窗口末端向前寻找
    # 逗号、冒号等较弱边界；实在没有标点时才硬切。
    sentences = [part.strip() for part in STRONG_BOUNDARY.split(cleaned) if part.strip()]
    pieces: list[str] = []
    for sentence in sentences:
        remaining = sentence
        while len(remaining) > max_chars:
            lower_bound = max_chars // 2
            cut_at = max(
                (remaining.rfind(mark, lower_bound, max_chars + 1) + 1 for mark in WEAK_BOUNDARY_CHARACTERS),
                default=0,
            )
            if cut_at <= 0:
                cut_at = max_chars
            pieces.append(remaining[:cut_at].strip())
            remaining = remaining[cut_at:].strip()
        if remaining:
            pieces.append(remaining)
    return pieces


def _logical_passages(lines: list[str], max_chars: int, overlap: int) -> list[str]:
    """把段落/句子装入窗口，并用完整逻辑单元做重叠。

    ``overlap`` 在这里是“最多希望复用多少字符”的软上限。为了不让下一块
    从半句话开始，只复用能够完整放入该上限的末尾句子或段落；如果没有
    合适的完整单元，就宁可不重叠。法律文本的可读边界比机械凑够 80 个字
    更重要。
    """

    units = [piece for line in lines for piece in _split_oversized_unit(line, max_chars)]
    if not units:
        return []

    passages: list[str] = []
    current: list[str] = []

    def joined_length(parts: list[str]) -> int:
        return sum(len(part) for part in parts) + max(0, len(parts) - 1)

    for unit in units:
        if current and joined_length([*current, unit]) > max_chars:
            passages.append("\n".join(current))
            overlap_units: list[str] = []
            for previous in reversed(current):
                candidate = [previous, *overlap_units]
                if joined_length(candidate) > overlap:
                    break
                overlap_units = candidate
            current = overlap_units
        if current and joined_length([*current, unit]) > max_chars:
            # 极端情况下重叠内容会挤不下当前单元，舍弃重叠即可；正文单元
            # 本身不会丢失，因为它马上会被加入新块。
            current = []
        current.append(unit)

    if current:
        passages.append("\n".join(current))
    return passages


def _context_lines(document_title: str, block: MarkdownBlock) -> list[str]:
    """生成每个子块都会重复的检索上下文。"""

    lines = [f"《{document_title}》"] if document_title else []
    for title in block.section_path:
        if title and title not in lines:
            lines.append(title)
    if block.structure_type == "preamble" and block.title not in lines:
        lines.append(block.title)
    return lines


def build_markdown_nodes(
    markdown_path: Path,
    document: SourceDocument,
    chunk_size: int,
    overlap: int,
) -> tuple[list[TextChunk], list[TextChunk]]:
    """一次解析同时构建完整父块和细粒度子块。

    返回值顺序为 ``(parents, children)``。父块保留一整条法律条文或一个
    普通章节，不参与相似度搜索；子块受 ``chunk_size`` 限制并参与检索。
    父子使用同一个稳定 ``parent_id``，在线阶段无需重新解析 Markdown。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    document_title, blocks = read_markdown_blocks(markdown_path)
    source_format = Path(document.raw_file).suffix.lower().lstrip(".")
    parents: list[TextChunk] = []
    children: list[TextChunk] = []
    type_counters = {"article": 0, "section": 0, "preamble": 0}

    for block in blocks:
        block_index = type_counters[block.structure_type]
        type_counters[block.structure_type] += 1
        parent_id = f"{document.document_id}-{block.structure_type}-{block_index:04d}"
        parent_article_id = parent_id if block.structure_type == "article" else ""
        parent_section_id = parent_id if block.structure_type != "article" else ""
        context_lines = _context_lines(document_title or document.title, block)
        context = "\n".join(context_lines)
        parent_body = block.body
        parent_text = f"{context}\n{parent_body}" if context else parent_body
        page_values = sorted(
            {segment.page_number for segment in block.segments if segment.page_number is not None}
        )
        parent_page_number = page_values[0] if len(page_values) == 1 else None
        parent_page_start = page_values[0] if page_values else None
        parent_page_end = page_values[-1] if page_values else None
        parents.append(
            TextChunk(
                chunk_id=parent_id,
                text=parent_text,
                chunk_index=len(parents),
                document=document,
                markdown_file=markdown_path.name,
                node_type="parent",
                chapter=block.chapter,
                article_number=block.article_number,
                structure_type=block.structure_type,
                section_title=block.title,
                section_path=" > ".join(block.section_path),
                parent_article_id=parent_article_id,
                parent_section_id=parent_section_id,
                child_index=-1,
                page_number=parent_page_number,
                page_start=parent_page_start,
                page_end=parent_page_end,
                page_numbers=",".join(str(page) for page in page_values),
                source_format=source_format,
            )
        )

        body_budget = chunk_size - len(context) - (1 if context else 0)
        if body_budget <= 0:
            raise ValueError(
                f"结构标题已达到 chunk_size，无法保留正文: {markdown_path.name} -> {block.section_path}"
            )

        child_index = 0
        for page_number, page_lines in _group_segments_by_page(block):
            # overlap 不能大于正文预算；标题很长时自动缩小重叠，但不改变
            # 全局配置，也不跨 PDF 页复用正文。
            page_overlap = min(overlap, max(0, body_budget - 1))
            for body in _logical_passages(page_lines, body_budget, page_overlap):
                text = f"{context}\n{body}" if context else body
                chunk_id = f"{parent_id}-part-{child_index:04d}"
                children.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=text,
                        chunk_index=len(children),
                        document=document,
                        markdown_file=markdown_path.name,
                        node_type="child",
                        chapter=block.chapter,
                        article_number=block.article_number,
                        structure_type=block.structure_type,
                        section_title=block.title,
                        section_path=" > ".join(block.section_path),
                        parent_article_id=parent_article_id,
                        parent_section_id=parent_section_id,
                        child_index=child_index,
                        page_number=page_number,
                        page_start=page_number,
                        page_end=page_number,
                        page_numbers=str(page_number or ""),
                        source_format=source_format,
                    )
                )
                child_index += 1
    return parents, children


def build_markdown_chunks(
    markdown_path: Path,
    document: SourceDocument,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    """兼容旧调用方：只返回参与检索的细粒度子块。"""

    _, children = build_markdown_nodes(markdown_path, document, chunk_size, overlap)
    return children


def build_chunks(text: str, document: SourceDocument, chunk_size: int, overlap: int) -> list[TextChunk]:
    """保留最初教学示例使用的纯文本固定窗口接口。"""

    passages = split_fixed_characters(text, chunk_size, overlap)
    return [
        TextChunk(
            chunk_id=f"{document.document_id}-chunk-{index:04d}",
            text=passage,
            chunk_index=index,
            document=document,
        )
        for index, passage in enumerate(passages)
    ]
