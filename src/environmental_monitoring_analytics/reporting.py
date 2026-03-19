import argparse
import csv
import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from html import escape
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "station_observations.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_COLUMNS = [
  "station_id",
  "station_name",
  "category",
  "region",
  "observed_at",
  "status",
  "alert_score",
  "reading_value",
]
STATUS_ALERT_SCORES = {
  "alert": 1.0,
  "normal": 0.25,
  "offline": 0.05,
}


def _data_path(data_path: Path | None = None, csv_path: Path | None = None) -> Path:
  return data_path or csv_path or DATA_PATH


def _coalesce(*values: object) -> object | None:
  for value in values:
    if value is not None:
      return value
  return None


def _extract_feature_lookup(payload: object) -> dict[str, dict[str, str]]:
  if not isinstance(payload, dict):
    return {}

  raw_features = payload.get("features", [])
  if isinstance(raw_features, dict):
    raw_features = raw_features.get("features", [])
  if not isinstance(raw_features, list):
    return {}

  feature_lookup: dict[str, dict[str, str]] = {}
  for feature in raw_features:
    if not isinstance(feature, dict):
      continue
    properties = feature.get("properties", feature)
    if not isinstance(properties, dict):
      continue
    feature_id = _coalesce(
      properties.get("featureId"),
      properties.get("feature_id"),
      properties.get("stationId"),
      properties.get("station_id"),
    )
    if feature_id is None:
      continue
    feature_lookup[str(feature_id)] = {
      "station_name": str(_coalesce(properties.get("name"), properties.get("stationName"), "")),
      "category": str(_coalesce(properties.get("category"), "")),
      "region": str(_coalesce(properties.get("region"), "")),
    }
  return feature_lookup


def _extract_observations(payload: object) -> list[dict[str, object]]:
  if isinstance(payload, list):
    return [item for item in payload if isinstance(item, dict)]
  if not isinstance(payload, dict):
    raise ValueError("Unsupported input payload; expected a list or object.")

  raw_observations = payload.get("observations", [])
  if isinstance(raw_observations, dict):
    raw_observations = raw_observations.get("observations", [])
  if not isinstance(raw_observations, list):
    raise ValueError("Input payload does not contain an observations list.")
  return [item for item in raw_observations if isinstance(item, dict)]


def _fallback_alert_score(status: str) -> float:
  return STATUS_ALERT_SCORES.get(status.lower(), 0.0)


def _normalize_snapshot_rows(data_path: Path) -> list[dict[str, object]]:
  payload = json.loads(data_path.read_text(encoding="utf-8"))
  feature_lookup = _extract_feature_lookup(payload)
  observations = _extract_observations(payload)

  rows: list[dict[str, object]] = []
  for observation in observations:
    station_id = _coalesce(
      observation.get("station_id"),
      observation.get("stationId"),
      observation.get("feature_id"),
      observation.get("featureId"),
    )
    observed_at = _coalesce(observation.get("observed_at"), observation.get("observedAt"))
    status = _coalesce(observation.get("status"), "normal")
    reading_value = _coalesce(
      observation.get("reading_value"),
      observation.get("readingValue"),
      observation.get("value"),
    )
    alert_score = _coalesce(observation.get("alert_score"), observation.get("alertScore"))
    feature_details = feature_lookup.get(str(station_id), {})

    row = {
      "station_id": station_id,
      "station_name": _coalesce(
        observation.get("station_name"),
        observation.get("stationName"),
        feature_details.get("station_name"),
      ),
      "category": _coalesce(observation.get("category"), feature_details.get("category")),
      "region": _coalesce(observation.get("region"), feature_details.get("region")),
      "observed_at": observed_at,
      "status": status,
      "alert_score": alert_score if alert_score is not None else _fallback_alert_score(str(status)),
      "reading_value": reading_value,
    }

    missing_fields = [column for column, value in row.items() if value is None]
    if missing_fields:
      raise ValueError(
        f"Input snapshot {data_path.name} is missing required fields: {', '.join(missing_fields)}"
      )
    rows.append(row)

  if not rows:
    raise ValueError(f"Input snapshot {data_path.name} does not contain any observations.")
  return rows


@contextmanager
def _normalized_csv_path(data_path: Path | None = None, csv_path: Path | None = None) -> Iterator[Path]:
  resolved_path = _data_path(data_path=data_path, csv_path=csv_path)
  if resolved_path.suffix.lower() == ".csv":
    yield resolved_path
    return

  rows = _normalize_snapshot_rows(resolved_path)
  with tempfile.TemporaryDirectory() as temp_dir:
    normalized_path = Path(temp_dir) / "normalized_observations.csv"
    with normalized_path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
      writer.writeheader()
      writer.writerows(rows)
    yield normalized_path


def _trend_direction(delta: float, tolerance: float = 1e-9) -> str:
    if delta > tolerance:
        return "worsening"
    if delta < -tolerance:
        return "improving"
    return "steady"


def _format_timestamp(value: str | None) -> str:
    return value or "n/a"


def compute_summary(data_path: Path | None = None, csv_path: Path | None = None) -> dict:
    with _normalized_csv_path(data_path=data_path, csv_path=csv_path) as path:
        csv_literal = str(path).replace("\\", "/").replace("'", "''")
        connection = duckdb.connect(database=":memory:")
        connection.execute(
            f"""
            CREATE VIEW observations AS
            SELECT *, strptime(observed_at, '%Y-%m-%dT%H:%M:%SZ') AS observed_ts
            FROM read_csv(
                '{csv_literal}',
                header=true,
                columns={{
                    'station_id': 'VARCHAR',
                    'station_name': 'VARCHAR',
                    'category': 'VARCHAR',
                    'region': 'VARCHAR',
                    'observed_at': 'VARCHAR',
                    'status': 'VARCHAR',
                    'alert_score': 'DOUBLE',
                    'reading_value': 'DOUBLE'
                }}
            )
            """
        )

        total_observations = connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        alert_observations = connection.execute(
            "SELECT COUNT(*) FROM observations WHERE status = 'alert'"
        ).fetchone()[0]
        average_alert_score = connection.execute(
            "SELECT ROUND(AVG(alert_score), 2) FROM observations"
        ).fetchone()[0]

        regional_alerts = connection.execute(
            """
            SELECT region, COUNT(*) AS alerts
            FROM observations
            WHERE status = 'alert'
            GROUP BY region
            ORDER BY alerts DESC, region ASC
            """
        ).fetchall()

        latest_alerts = connection.execute(
            """
            SELECT station_name, region, category, observed_at, alert_score
            FROM observations
            WHERE status = 'alert'
            ORDER BY observed_ts DESC
            LIMIT 3
            """
        ).fetchall()

        latest_timestamp = connection.execute("SELECT MAX(observed_ts) FROM observations").fetchone()[0]
        trend_windows = connection.execute(
            """
            WITH bounds AS (
              SELECT
                MAX(observed_ts) AS latest_ts,
                MAX(observed_ts) - INTERVAL '2 hours' AS recent_start,
                MAX(observed_ts) - INTERVAL '4 hours' AS previous_start
              FROM observations
            ),
            windowed AS (
              SELECT
                CASE
                  WHEN observed_ts > recent_start AND observed_ts <= latest_ts THEN 'recent'
                  WHEN observed_ts > previous_start AND observed_ts <= recent_start THEN 'previous'
                  ELSE NULL
                END AS window_name,
                observed_at,
                status,
                alert_score
              FROM observations, bounds
            )
            SELECT
              window_name,
              COUNT(*) AS total_observations,
              SUM(CASE WHEN status = 'alert' THEN 1 ELSE 0 END) AS alert_observations,
              ROUND(AVG(alert_score), 2) AS average_alert_score,
              MIN(observed_at) AS earliest_observed_at,
              MAX(observed_at) AS latest_observed_at
            FROM windowed
            WHERE window_name IS NOT NULL
            GROUP BY window_name
            ORDER BY window_name DESC
            """
        ).fetchall()

        regional_trends = connection.execute(
            """
            WITH bounds AS (
              SELECT
                MAX(observed_ts) AS latest_ts,
                MAX(observed_ts) - INTERVAL '2 hours' AS recent_start,
                MAX(observed_ts) - INTERVAL '4 hours' AS previous_start
              FROM observations
            ),
            recent AS (
              SELECT region, COUNT(*) AS alerts
              FROM observations, bounds
              WHERE status = 'alert'
                AND observed_ts > recent_start
                AND observed_ts <= latest_ts
              GROUP BY region
            ),
            previous AS (
              SELECT region, COUNT(*) AS alerts
              FROM observations, bounds
              WHERE status = 'alert'
                AND observed_ts > previous_start
                AND observed_ts <= recent_start
              GROUP BY region
            )
            SELECT
              COALESCE(recent.region, previous.region) AS region,
              COALESCE(recent.alerts, 0) AS recent_alerts,
              COALESCE(previous.alerts, 0) AS previous_alerts,
              COALESCE(recent.alerts, 0) - COALESCE(previous.alerts, 0) AS alert_delta
            FROM recent
            FULL OUTER JOIN previous ON recent.region = previous.region
            ORDER BY alert_delta DESC, region ASC
            """
        ).fetchall()
        connection.close()

        trend_by_window = {
            row[0]: {
                "total_observations": row[1],
                "alert_observations": row[2],
                "alert_rate": round((row[2] / row[1]), 4) if row[1] else 0.0,
                "average_alert_score": row[3],
                "earliest_observed_at": row[4],
                "latest_observed_at": row[5],
            }
            for row in trend_windows
        }
        recent_window = trend_by_window.get(
            "recent",
            {
                "total_observations": 0,
                "alert_observations": 0,
                "alert_rate": 0.0,
                "average_alert_score": None,
                "earliest_observed_at": None,
                "latest_observed_at": None,
            },
        )
        previous_window = trend_by_window.get(
            "previous",
            {
                "total_observations": 0,
                "alert_observations": 0,
                "alert_rate": 0.0,
                "average_alert_score": None,
                "earliest_observed_at": None,
                "latest_observed_at": None,
            },
        )

        average_delta = round(
            (recent_window["average_alert_score"] or 0.0) - (previous_window["average_alert_score"] or 0.0),
            2,
        )
        alert_rate_delta = round(recent_window["alert_rate"] - previous_window["alert_rate"], 4)

        return {
            "total_observations": total_observations,
            "alert_observations": alert_observations,
            "alert_rate": round(alert_observations / total_observations, 4) if total_observations else 0.0,
            "average_alert_score": average_alert_score,
            "regional_alerts": regional_alerts,
            "latest_alerts": latest_alerts,
            "time_window_trends": {
                "window_hours": 2,
                "latest_timestamp": str(latest_timestamp) if latest_timestamp is not None else None,
                "recent": recent_window,
                "previous": previous_window,
                "alert_rate_delta": alert_rate_delta,
                "average_alert_score_delta": average_delta,
                "direction": _trend_direction(alert_rate_delta),
                "regional_changes": regional_trends,
            },
        }


def build_markdown_report(data_path: Path | None = None, csv_path: Path | None = None) -> str:
    summary = compute_summary(data_path=data_path, csv_path=csv_path)
    trends = summary["time_window_trends"]

    regional_lines = "\n".join(
        f"- {region}: {alerts} alert observations" for region, alerts in summary["regional_alerts"]
    )
    latest_lines = "\n".join(
        f"- {station} ({region}, {category}) at {observed_at} with alert score {alert_score}"
        for station, region, category, observed_at, alert_score in summary["latest_alerts"]
    )
    regional_trend_lines = "\n".join(
        f"- {region}: {recent_alerts} alerts in the recent window vs {previous_alerts} in the previous window ({alert_delta:+d})"
        for region, recent_alerts, previous_alerts, alert_delta in trends["regional_changes"]
    )

    return f"""# Monitoring Operations Brief

## Summary

- Total observations: {summary['total_observations']}
- Alert observations: {summary['alert_observations']}
- Alert rate: {summary['alert_rate']:.2%}
- Average alert score: {summary['average_alert_score']}

## Regional Alert Load

{regional_lines}

## Trend Window

- Window size: {trends['window_hours']} hours
- Recent window: {_format_timestamp(trends['recent']['earliest_observed_at'])} to {_format_timestamp(trends['recent']['latest_observed_at'])}
- Previous window: {_format_timestamp(trends['previous']['earliest_observed_at'])} to {_format_timestamp(trends['previous']['latest_observed_at'])}
- Recent alert rate: {trends['recent']['alert_rate']:.2%} ({trends['recent']['alert_observations']} of {trends['recent']['total_observations']})
- Previous alert rate: {trends['previous']['alert_rate']:.2%} ({trends['previous']['alert_observations']} of {trends['previous']['total_observations']})
- Alert-rate direction: {trends['direction']} ({trends['alert_rate_delta']:+.2%})
- Average alert-score delta: {trends['average_alert_score_delta']:+.2f}

## Regional Trend Shift

{regional_trend_lines}

## Latest Alert Stations

{latest_lines}
"""


def build_html_report(data_path: Path | None = None, csv_path: Path | None = None) -> str:
    summary = compute_summary(data_path=data_path, csv_path=csv_path)
    trends = summary["time_window_trends"]
    max_alerts = max((alerts for _, alerts in summary["regional_alerts"]), default=1)

    def bar_width(alerts: int) -> int:
        return max(12, round((alerts / max_alerts) * 100))

    regional_cards = "\n".join(
        f"""
        <div class="bar-group bar-width-{bar_width(alerts)}">
          <div class="bar-label"><span>{escape(region)}</span><strong>{alerts}</strong></div>
          <div class="bar-track"><div class="bar-fill"></div></div>
        </div>
        """.strip()
        for region, alerts in summary["regional_alerts"]
    )
    latest_cards = "\n".join(
        f"<li>{escape(station)} ({escape(region)}, {escape(category)}) at {escape(observed_at)} with alert score {alert_score}</li>"
        for station, region, category, observed_at, alert_score in summary["latest_alerts"]
    )
    trend_cards = "\n".join(
        f"<li><strong>{escape(region)}</strong>: {recent_alerts} alerts in the recent window vs {previous_alerts} previously ({alert_delta:+d})</li>"
        for region, recent_alerts, previous_alerts, alert_delta in trends["regional_changes"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Monitoring Operations Brief</title>
    <style>
      body {{ font-family: Georgia, "Times New Roman", serif; margin: 0; background: #f6f2e8; color: #1f2a25; }}
      main {{ max-width: 960px; margin: 0 auto; padding: 32px 20px 48px; }}
      .hero, .card {{ background: rgba(255,255,255,0.82); border: 1px solid rgba(31,42,37,0.08); border-radius: 22px; padding: 24px; box-shadow: 0 18px 40px rgba(47,56,50,0.08); }}
      .hero {{ margin-bottom: 18px; }}
      .eyebrow {{ text-transform: uppercase; letter-spacing: 0.18em; font-size: 0.76rem; color: #5f6d66; margin: 0 0 10px; }}
      .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 18px; }}
      .metric {{ font-size: 2rem; margin: 8px 0; }}
      .bars, .alerts {{ display: grid; gap: 12px; }}
      .trend-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 18px 0; }}
      .bar-label {{ display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.95rem; }}
      .bar-track {{ height: 12px; border-radius: 999px; background: rgba(73,97,109,0.12); overflow: hidden; }}
      .bar-fill {{ height: 100%; background: linear-gradient(90deg, #49616d, #d4a85f); }}
      .bar-width-12 .bar-fill {{ width: 12%; }}
      .bar-width-50 .bar-fill {{ width: 50%; }}
      .bar-width-100 .bar-fill {{ width: 100%; }}
      .two-col {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }}
      .trend-badge {{ display: inline-flex; padding: 6px 10px; border-radius: 999px; background: rgba(212,168,95,0.16); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; }}
      ul {{ margin: 0; padding-left: 18px; }}
      li {{ margin-bottom: 10px; }}
      @media (max-width: 800px) {{ .grid, .two-col, .trend-grid {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <p class="eyebrow">Generated Report</p>
        <h1>Monitoring Operations Brief</h1>
        <p>Operational HTML summary for alert load, latest priority stations, and regional monitoring pressure.</p>
      </section>
      <section class="grid">
        <article class="card"><p class="eyebrow">Observations</p><div class="metric">{summary['total_observations']}</div><p>Total observations processed.</p></article>
        <article class="card"><p class="eyebrow">Alert Rate</p><div class="metric">{summary['alert_rate']:.2%}</div><p>Observations currently flagged as alerts.</p></article>
        <article class="card"><p class="eyebrow">Avg Alert Score</p><div class="metric">{summary['average_alert_score']}</div><p>Average alert score across the dataset.</p></article>
      </section>
      <section class="trend-grid">
        <article class="card">
          <p class="eyebrow">Trend Window</p>
          <p class="trend-badge">{escape(trends['direction'])}</p>
          <p>Recent window: {_format_timestamp(trends['recent']['earliest_observed_at'])} to {_format_timestamp(trends['recent']['latest_observed_at'])}</p>
          <p>Previous window: {_format_timestamp(trends['previous']['earliest_observed_at'])} to {_format_timestamp(trends['previous']['latest_observed_at'])}</p>
          <p><strong>Recent alert rate:</strong> {trends['recent']['alert_rate']:.2%}</p>
          <p><strong>Previous alert rate:</strong> {trends['previous']['alert_rate']:.2%}</p>
          <p><strong>Alert-rate delta:</strong> {trends['alert_rate_delta']:+.2%}</p>
          <p><strong>Avg alert-score delta:</strong> {trends['average_alert_score_delta']:+.2f}</p>
        </article>
        <article class="card">
          <p class="eyebrow">Regional Trend Shift</p>
          <ul class="alerts">{trend_cards}</ul>
        </article>
      </section>
      <section class="two-col">
        <article class="card">
          <p class="eyebrow">Regional Alert Load</p>
          <div class="bars">{regional_cards}</div>
        </article>
        <article class="card">
          <p class="eyebrow">Latest Alert Stations</p>
          <ul class="alerts">{latest_cards}</ul>
        </article>
      </section>
    </main>
  </body>
</html>
"""


def export_reports(
    output_dir: Path | None = None,
    data_path: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, Path]:
    target_dir = output_dir or OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = target_dir / "monitoring-operations-brief.md"
    html_path = target_dir / "monitoring-operations-brief.html"
    markdown_path.write_text(build_markdown_report(data_path=data_path, csv_path=csv_path), encoding="utf-8")
    html_path.write_text(build_html_report(data_path=data_path, csv_path=csv_path), encoding="utf-8")
    return {"markdown": markdown_path, "html": html_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate monitoring operations reports from CSV or API snapshot input.")
    parser.add_argument("--input", type=Path, default=None, help="Optional path to a CSV dataset or API-derived JSON snapshot.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for generated markdown and HTML reports.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(build_markdown_report(data_path=args.input))
    output_paths = export_reports(output_dir=args.output_dir, data_path=args.input)
    print(f"\nWrote {output_paths['markdown']} and {output_paths['html']}")


if __name__ == "__main__":
    main()