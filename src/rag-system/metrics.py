from pathlib import Path

import yaml


CATALOG_PATH = Path(__file__).parent / "metrics_catalog.yaml"


class MetricsCatalog:
    """Loads metric definitions from YAML and provides them as LLM context."""

    def __init__(self, catalog_path: str | Path | None = None):
        self._path = Path(catalog_path) if catalog_path else CATALOG_PATH
        self._metrics: dict[str, dict] = {}
        self._load()

    def _load(self):
        with open(self._path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._metrics = data.get("metrics", {})

    def reload(self):
        self._load()

    @property
    def metric_ids(self) -> list[str]:
        return list(self._metrics.keys())

    def get(self, metric_id: str) -> dict | None:
        return self._metrics.get(metric_id)

    def build_all_context(self) -> str:
        """Build a full context string of all metric definitions for the SQL prompt."""
        if not self._metrics:
            return ""

        parts = []
        for mid, m in self._metrics.items():
            section = [
                f"Metric: {m['name']} ({mid})",
                f"  Description: {m['description'].strip()}",
                f"  Formula: {m['formula']}",
                f"  Unit: {m['unit']}",
                f"  Tables: {', '.join(m['tables'])}",
                f"  Steps:",
            ]
            for step in m.get("steps", []):
                section.append(f"    Step {step['step']}: {step['action']}")
                if step.get("uses"):
                    section.append(f"      Uses: {step['uses']}")
                if step.get("note"):
                    section.append(f"      Note: {step['note']}")

            if m.get("parameters"):
                section.append("  Parameters:")
                for p in m["parameters"]:
                    opt = " (optional)" if p.get("optional") else ""
                    section.append(f"    {p['name']}: {p['description']}{opt} [default={p.get('default', 'N/A')}]")

            if m.get("thresholds"):
                section.append("  Thresholds:")
                for level, t in m["thresholds"].items():
                    section.append(f"    {level}: {t.get('label', '')}")

            parts.append("\n".join(section))

        return "\n\n".join(parts)
