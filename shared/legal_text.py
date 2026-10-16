"""中文检索分词与法规引用解析；不依赖分词模型，也不猜测法规内容。"""

from __future__ import annotations

import re

ARTICLE_PATTERN = re.compile(r"第\s*([一二三四五六七八九十百千万零〇两0-9]+)\s*条")
TITLE_PATTERN = re.compile(r"《([^》]+)》")
DOCUMENT_NUMBER_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,10}(?:〔|\[)\d{4}(?:〕|\])\s*\d+号|(?:中华人民共和国)?(?:财政部|国务院|国家发展和改革委员会|国家发展改革委)(?:令第?\d+号|\d+号令)")


def compact(text: str) -> str:
    return re.sub(r"[\s，。！？；：、（）()《》\[\]〔〕\"'·-]", "", text).lower()


def article_value(text: str) -> int | None:
    match = ARTICLE_PATTERN.search(text)
    if not match:
        return None
    value = match.group(1)
    if value.isdigit():
        return int(value)
    digits = dict(zip("零〇一二三四五六七八九两", (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2)))
    total, number = 0, 0
    for char in value:
        if char in digits:
            number = digits[char]
        elif char in "十百千万":
            unit = {"十": 10, "百": 100, "千": 1000, "万": 10000}[char]
            total += (number or 1) * unit
            number = 0
    return total + number


def title_matches(query: str, title: str) -> bool:
    """只允许全称及去掉“中华人民共和国”的等值简称，避免法与实施条例混淆。"""
    return compact(query).removeprefix("中华人民共和国") == compact(title).removeprefix("中华人民共和国")


def number_key(value: str) -> str:
    value = compact(value).removeprefix("中华人民共和国")
    value = re.sub(r"^(?:以及|和|与|及|请问|查询|根据|依照|那么|那|再查)+", "", value)
    match = re.fullmatch(r"(.+?)(?:令第?(\d+)号|(\d+)号令)", value)
    return f"{match[1]}令{match[2] or match[3]}号" if match else value


def tokens(text: str) -> list[str]:
    """中文用二元字串，英文/数字用完整词；FTS 输入全部来自本函数。"""
    result: list[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            result.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            result.append(part)
    return result


def references(question: str) -> list[dict[str, str]]:
    """按文本位置配对法规与条号；同一标题可对应多个条号。"""
    titles = [(m.start(), m.end(), m.group(1)) for m in TITLE_PATTERN.finditer(question)]
    pattern = r"((?:中华人民共和国)?[\u4e00-\u9fff]{2,50}?(?:法实施条例|实施条例|条例|办法|规定|法))(?=\s*第[一二三四五六七八九十百千万零〇两0-9]+条)"
    for m in re.finditer(pattern, question):
        if any(start <= m.start() < end for start, end, _ in titles):
            continue
        candidate = re.sub(r"^第[一二三四五六七八九十百千万零〇两0-9]+条(?:以及|和|与|及)?", "", m.group(1))
        candidate = re.split(r"以及|和|与|及", candidate)[-1] if re.search(r"条(?:以及|和|与|及)", candidate) else candidate
        title = re.sub(r"^(?:请问|查询|请查|根据|依照|关于|帮我查|请展示|请提供|请查阅|请给出|给我查|请解释|那么|那|再查|再看)+", "", candidate)
        titles.append((m.start(), m.end(), title))
    titles.sort()
    numbers = [(m.start(), m.end(), m.group(0)) for m in DOCUMENT_NUMBER_PATTERN.finditer(question)]
    result: list[dict[str, str]] = []
    for match in ARTICLE_PATTERN.finditer(question):
        preceding = [row for row in titles if row[0] < match.start()]
        title = preceding[-1][2] if preceding else ""
        preceding_numbers = [row for row in numbers if row[1] <= match.start() and (not preceding or row[0] >= preceding[-1][0])]
        number = preceding_numbers[-1][2] if preceding_numbers else ""
        if number and preceding and ARTICLE_PATTERN.search(question[preceding[-1][1]:preceding_numbers[-1][0]]):
            title = ""  # 新文号已经开启另一处引用，不能继承前一法条的名称。
        result.append({"document_title": title, "article_number": match.group(0).replace(" ", ""), "document_number": number})
    if not result:
        result = [{"document_title": title, "article_number": "", "document_number": ""} for _, _, title in titles]
    for _, _, number in numbers:
        if not any(row["document_number"] == number for row in result):
            result.append({"document_title": "", "article_number": "", "document_number": number})
    return list({tuple(row.values()): row for row in result}.values())
