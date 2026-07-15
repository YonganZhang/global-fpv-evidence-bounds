#!/usr/bin/env python3
"""Acceptance checks for the v118 six-author/open-release manuscript."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "paper/draft"
FIGURES = ROOT / "paper/figures/v2026-07-15_v117_rebuild/main"
REPORTS = ROOT / "paper/draft/review_versions/2026-07-15_v118_six_author_open_release/reports"
ACTIVE = [
    "abstract.tex", "frontmatter.tex", "introduction.tex", "methods.tex",
    "results_discussion.tex", "conclusion.tex", "main_figures.tex",
    "supplementary.tex", "backmatter.tex", "cover_letter.tex", "highlights.txt",
]


def png_dimensions(path: Path) -> tuple[int, int] | None:
    header = path.read_bytes()[:24] if path.exists() else b""
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def plain_tex(text: str) -> str:
    # Strip real LaTeX comments without treating escaped percentage signs as
    # comment starts. The abstract is intentionally one source line, so the
    # latter would otherwise discard everything after its first ``\%``.
    text = re.sub(r"(?<!\\)%.*", " ", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    texts = {name: (DRAFT / name).read_text(encoding="utf-8") for name in ACTIVE}
    joined = "\n".join(texts.values())
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    gate = json.loads((ROOT / "_outputs/v117/reports/consolidated_validation_gate_v117.json").read_text())
    checks["scientific_gate_13_of_13"] = gate.get("status") == "pass" and gate.get("passed") == 13

    forbidden = ["58,031", "58031", "31,383", "31383", "74,031", "74031", "65,872", "65872",
                 "16,579", "16579", "10,870", "10870", "4,128", "4128", "4,743", "4743",
                 "2,869", "2869", "3,359", "3359", "1,569", "1569", "3,458", "3458",
                 "1,083", "1083", "2,457", "2457", "198,731", "198731"]
    hits = {value: [name for name, text in texts.items() if value in text] for value in forbidden}
    hits = {key: value for key, value in hits.items() if value}
    checks["no_superseded_endpoint_numbers"] = not hits
    details["superseded_number_hits"] = hits

    required = ["199{,}976", "16{,}713", "10{,}995", "4{,}190--4{,}805", "2{,}914--3{,}404",
                "1{,}595--3{,}508", "1{,}101--2{,}494"]
    checks["required_v117_display_values_present"] = all(value in joined for value in required)

    lower = joined.lower()
    checks["complete_l2_remains_unavailable"] = "complete" in lower and "l2" in lower and "unavailable" in lower
    checks["complete_l3_remains_unavailable"] = "complete deployable" in lower and "unavailable" in lower
    checks["glev_period_stated"] = "1991--2018" in joined and "2019--2020" in joined
    checks["powerline_capacity_boundary_stated"] = "residual-capacity" in lower or "residual interconnection capacity" in lower
    checks["depth_boundary_stated"] = "not depth at a candidate array or anchor" in lower or "not site or anchor depth" in lower
    checks["lcoe_boundary_stated"] = "not a project quote" in lower or "neither a project quote" in lower

    author_order = ["Yongan Zhang", "Yujie Jiang", "Xiaoyuan Li", "Zhiling Guo", "Yuntian Chen", "Jinyue Yan"]
    front_positions = [texts["frontmatter.tex"].find(name) for name in author_order]
    cover_positions = [texts["cover_letter.tex"].find(name) for name in author_order]
    details["frontmatter_author_positions"] = dict(zip(author_order, front_positions))
    details["cover_letter_author_positions"] = dict(zip(author_order, cover_positions))
    checks["six_author_order_frontmatter"] = all(position >= 0 for position in front_positions) and front_positions == sorted(front_positions)
    checks["six_author_order_cover_letter"] = all(position >= 0 for position in cover_positions) and cover_positions == sorted(cover_positions)
    checks["xiaoyuan_affiliations_match_yujie"] = (
        r"\author[polyu-beee,polyu-icen]{Yujie Jiang}" in texts["frontmatter.tex"]
        and r"\author[polyu-beee,polyu-icen]{Xiaoyuan Li}" in texts["frontmatter.tex"]
    )
    checks["xiaoyuan_credit_present"] = "Xiaoyuan Li: Investigation, Validation" in texts["backmatter.tex"]
    checks["public_repository_declared"] = (
        "https://github.com/YonganZhang/global-fpv-evidence-bounds" in texts["backmatter.tex"]
        and "v1.0.0-v118" in texts["backmatter.tex"]
    )
    placeholders = ["[username]", "to be issued upon acceptance", "persistent accession must be inserted", "repository DOI/accession"]
    placeholder_hits = [value for value in placeholders if value.lower() in joined.lower()]
    details["submission_placeholder_hits"] = placeholder_hits
    checks["no_repository_placeholders"] = not placeholder_hits
    checks["no_unverified_default_grants"] = "ZD2019-183-004" not in joined and "20CX05019A" not in joined
    checks["funding_control_is_explicit"] = "no verified funder or grant identifier" in lower

    open_gate_path = ROOT / "_outputs/v118/open_repository/OPEN_RELEASE_VALIDATION.json"
    open_gate = json.loads(open_gate_path.read_text(encoding="utf-8")) if open_gate_path.exists() else {}
    checks["open_release_gate_12_of_12"] = open_gate.get("status") == "pass" and open_gate.get("passed") == 12

    abstract = plain_tex(texts["abstract.tex"])
    abstract_words = len(re.findall(r"\b[\w$%.-]+\b", abstract))
    details["abstract_word_count_approx"] = abstract_words
    checks["abstract_under_250_words"] = abstract_words <= 250

    highlights = [line.strip() for line in texts["highlights.txt"].splitlines() if line.strip()]
    details["highlight_characters"] = [len(line) for line in highlights]
    checks["highlights_count_3_to_5"] = 3 <= len(highlights) <= 5
    checks["highlights_max_85_characters"] = all(len(line) <= 85 for line in highlights)

    cited: set[str] = set()
    for text in texts.values():
        for block in re.findall(r"\\cite[tp]?\{([^}]+)\}", text):
            cited.update(item.strip() for item in block.split(","))
    bib = (DRAFT / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
    missing_citations = sorted(cited - keys)
    details["missing_citation_keys"] = missing_citations
    checks["all_citation_keys_resolve"] = not missing_citations

    required_figures = [f"Fig{i}.{ext}" for i in range(1, 6) for ext in ["png", "pdf"]]
    required_figures += [f"FigS{i}.{ext}" for i in range(1, 3) for ext in ["png", "pdf"]]
    required_figures += [f"graphical_abstract.{ext}" for ext in ["png", "pdf"]]
    missing_figures = [name for name in required_figures if not (FIGURES / name).exists()]
    details["missing_figures"] = missing_figures
    checks["all_figure_files_exist"] = not missing_figures
    graphical_abstract_dimensions = png_dimensions(FIGURES / "graphical_abstract.png")
    details["graphical_abstract_dimensions"] = graphical_abstract_dimensions
    checks["graphical_abstract_is_3000_by_1200"] = graphical_abstract_dimensions == (3000, 1200)
    rejected_stems = [
        "fig1_inventory", "fig2_geographic", "fig3_model_eval",
        "fig4_l2_evidence", "fig5_l3_uncertainty",
    ]
    checks["rejected_placeholder_figures_excluded"] = not any(
        (FIGURES / f"{stem}.{ext}").exists()
        for stem in rejected_stems
        for ext in ["png", "pdf"]
    )
    checks["no_header_only_figures_page"] = r"\section*{Figures}" not in texts["main_figures.tex"]

    legacy_merge = (ROOT / "src/data_prep/merge_waterbodies.py").read_text(encoding="utf-8")
    checks["legacy_proximity_merge_retired"] = (
        "raise RuntimeError" in legacy_merge
        and "Retired proximity-only merge" in legacy_merge
        and "rebuild_v117_scientific_revision.py" in legacy_merge
    )

    for name in ["main.log", "main_submission.log"]:
        log = (DRAFT / name).read_text(encoding="utf-8") if (DRAFT / name).exists() else "missing"
        checks[f"{name}_no_undefined_references"] = "undefined references" not in log.lower() and "undefined citations" not in log.lower()
        checks[f"{name}_no_overfull_boxes"] = "Overfull \\hbox" not in log

    checks["main_pdf_exists"] = (DRAFT / "main.pdf").exists()
    checks["submission_pdf_exists"] = (DRAFT / "main_submission.pdf").exists()
    checks["cover_letter_pdf_exists"] = (DRAFT / "cover_letter.pdf").exists()
    checks["highlights_pdf_exists"] = (DRAFT / "highlights.pdf").exists()
    checks["chinese_review_pdf_exists"] = (DRAFT / "main_中文版_v118_审阅稿.pdf").exists()

    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
        "details": details,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "manuscript_acceptance_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# v118 manuscript acceptance gate", "", f"Status: **{result['status'].upper()}** ({result['passed']}/{result['total']})", ""]
    lines.extend(f"- {'PASS' if ok else 'FAIL'}: `{name}`" for name, ok in checks.items())
    lines += ["", "## Details", "", "```json", json.dumps(details, indent=2), "```", ""]
    (REPORTS / "MANUSCRIPT_ACCEPTANCE_GATE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
