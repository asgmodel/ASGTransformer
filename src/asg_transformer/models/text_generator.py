from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from asg_transformer.core.catalog import CatalogItem, KnowledgeCatalog
from asg_transformer.models.duration_planner import PlannedStep
from asg_transformer.models.semantic_encoder import RankedItem


@dataclass(slots=True)
class GeneratedScenarioText:
    title: str
    executive_summary: str
    scenario_text: str


class GroundedTextGenerator:
    """Grounded professional text generator.

    It renders text only from catalog-backed entities selected by the encoder
    and planner. This avoids unsupported techniques and keeps output stable.
    An optional LLM polishing layer can be added later without changing the
    public model interface.
    """

    def __init__(self, catalog: KnowledgeCatalog) -> None:
        self.catalog = catalog
        self._technique_index = {item.label: item for item in catalog.techniques}

    def generate(
        self,
        input_text: str,
        steps: Sequence[PlannedStep],
        software: Sequence[RankedItem],
        groups: Sequence[RankedItem],
        language: str = "en",
    ) -> GeneratedScenarioText:
        if language.lower().startswith("ar"):
            return self._generate_ar(input_text, steps, software, groups)
        return self._generate_en(input_text, steps, software, groups)

    def _generate_en(
        self,
        input_text: str,
        steps: Sequence[PlannedStep],
        software: Sequence[RankedItem],
        groups: Sequence[RankedItem],
    ) -> GeneratedScenarioText:
        title = "ASG Grounded Cybersecurity Scenario"
        if not steps:
            summary = "No sufficiently relevant catalog-backed scenario could be produced."
            return GeneratedScenarioText(title, summary, summary)

        confidence = sum(step.combined_score for step in steps) / len(steps)
        total_duration = sum(step.duration_minutes for step in steps)
        summary = (
            f"A {len(steps)}-stage defensive simulation was generated from the supplied objective. "
            f"The plan spans approximately {total_duration} minutes and has an average grounded "
            f"confidence of {confidence:.2%}."
        )

        lines = [
            title,
            "",
            "Objective",
            input_text.strip(),
            "",
            "Executive Summary",
            summary,
            "",
            "Scenario Stages",
        ]
        for step in steps:
            item = self._technique_index.get(step.technique)
            description = item.description.strip() if item and item.description else "Catalog-backed technique."
            lines.extend(
                [
                    f"{step.order}. {step.tactic} — {step.technique}",
                    f"   Estimated duration: {step.duration_minutes} minutes",
                    f"   Rationale: {description}",
                    f"   Grounding score: {step.combined_score:.3f}",
                ]
            )

        if software:
            lines.extend(["", "Related Software", ", ".join(item.item.label for item in software)])
        if groups:
            lines.extend(["", "Related Threat Groups", ", ".join(item.item.label for item in groups)])

        lines.extend(
            [
                "",
                "Defensive Use",
                "Use this output for tabletop exercises, control validation, detection engineering, "
                "incident-response preparation, and authorized security training. Review every stage "
                "against the organization’s policies and environment before execution.",
            ]
        )
        return GeneratedScenarioText(title, summary, "\n".join(lines))

    def _generate_ar(
        self,
        input_text: str,
        steps: Sequence[PlannedStep],
        software: Sequence[RankedItem],
        groups: Sequence[RankedItem],
    ) -> GeneratedScenarioText:
        title = "سيناريو أمن سيبراني مؤسس على قاعدة ASG"
        if not steps:
            summary = "لم يتم العثور على مسار ذي صلة كافية داخل قاعدة المعرفة."
            return GeneratedScenarioText(title, summary, summary)

        confidence = sum(step.combined_score for step in steps) / len(steps)
        total_duration = sum(step.duration_minutes for step in steps)
        summary = (
            f"تم إنشاء محاكاة دفاعية من {len(steps)} مراحل بمدة تقديرية تبلغ {total_duration} دقيقة، "
            f"ومتوسط ثقة مؤسس على قاعدة المعرفة قدره {confidence:.2%}."
        )
        lines = [title, "", "الهدف", input_text.strip(), "", "الملخص التنفيذي", summary, "", "مراحل السيناريو"]
        for step in steps:
            item = self._technique_index.get(step.technique)
            description = item.description.strip() if item and item.description else "تقنية مرتبطة بقاعدة المعرفة."
            lines.extend(
                [
                    f"{step.order}. {step.tactic} — {step.technique}",
                    f"   المدة التقديرية: {step.duration_minutes} دقيقة",
                    f"   الوصف: {description}",
                    f"   درجة الربط: {step.combined_score:.3f}",
                ]
            )
        if software:
            lines.extend(["", "البرمجيات ذات الصلة", "، ".join(item.item.label for item in software)])
        if groups:
            lines.extend(["", "مجموعات التهديد ذات الصلة", "، ".join(item.item.label for item in groups)])
        lines.extend(
            [
                "",
                "الاستخدام الدفاعي",
                "يستخدم هذا الناتج في التمارين المكتبية، والتحقق من الضوابط، وهندسة الرصد، "
                "والاستعداد للاستجابة للحوادث، والتدريب الأمني المصرح به.",
            ]
        )
        return GeneratedScenarioText(title, summary, "\n".join(lines))
