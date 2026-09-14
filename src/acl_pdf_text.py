"""Спільний модуль для round 14 (PDF-пайплайн для 285 ACL-only оглядів,
docs/superpowers/specs/2026-09-09-acl-pdf-pipeline-design.md). Витягує
повний текст PDF і локалізує секцію списку літератури -- використовується
і mine_disagreement_from_acl_pdfs.py (повний текст), і
resolve_acl_pdf_citations.py (лише секція референсів).

Заголовок секції варіюється між venue ("References", "Bibliographical
References", "Language Resource References" -- підтверджено емпірично на
реальних PDF round 14) -- звідси толерантний до префікса regex, а не
точний збіг "References".

Control-байти (0x00-0x08, 0x0b, 0x0c, 0x0e-0x1f) прибираються тут, у ЄДИНОМУ
місці екстракції -- інколи з'являються замість мангленого математичного/
спецсимвольного глифа при PDF-екстракції (напр. "N choose 2" в
комбінаториці); лишений \\x00 у полі CSV призводить до того, що pandas
C-парсер зчитує це поле назад обрізаним точно на NUL. Раніше (round 14)
цей strip був лише в mine_disagreement_from_acl_pdfs.py -- перенесено сюди
(round 15, tech-debt fix #3), щоб і resolve_acl_pdf_citations.py, який
незалежно повторно екстрактує ті самі PDF для парсингу списку літератури,
успадкував той самий захист автоматично.
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

REFERENCES_HEADING_PATTERN = re.compile(r'\n([^\n]{0,30}\b[Rr]eferences\b[^\n]{0,10})\n')
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def extract_pdf_text(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    try:
        text = "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()
    return CONTROL_CHAR_PATTERN.sub("", text)


def find_references_section(text: str) -> str | None:
    """Останній збіг заголовка -- перший міг би трапитись у змісті/переліку
    розділів на початку статті, а не в самій секції референсів."""
    matches = list(REFERENCES_HEADING_PATTERN.finditer(text))
    if not matches:
        return None
    return text[matches[-1].end():]
