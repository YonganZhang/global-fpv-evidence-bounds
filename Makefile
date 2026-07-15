.PHONY: validate figures paper

validate:
	uv run python scripts/validate_release.py

figures:
	test -f _outputs/v117/data/fpv_reference_inventory_v117.parquet || (echo "Full licensed inventory missing; see REPRODUCIBILITY.md" && exit 2)
	uv run python src/figures/draw_v117_manuscript_figures.py

paper:
	cd paper/draft && latexmk -g -pdf -interaction=nonstopmode -halt-on-error main_submission.tex
	cd paper/draft && latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
