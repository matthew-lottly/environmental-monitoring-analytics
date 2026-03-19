from html import escape
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "station_observations.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _csv_path(csv_path: Path | None = None) -> Path:
    return csv_path or DATA_PATH


def _trend_direction(delta: float, tolerance: float = 1e-9) -> str:
    if delta > tolerance:
        return "worsening"
    if delta < -tolerance:
        return "improving"
    return "steady"


def _format_timestamp(value: str | None) -> str:
    return value or "n/a"


def compute_summary(csv_path: Path | None = None) -> dict:
    path = _csv_path(csv_path)
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
        "alert_rate": round(alert_observations / total_observations, 4),
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


def build_markdown_report(csv_path: Path | None = None) -> str:
    summary = compute_summary(csv_path)
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


def build_html_report(csv_path: Path | None = None) -> str:
    summary = compute_summary(csv_path)
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


def export_reports(output_dir: Path | None = None, csv_path: Path | None = None) -> dict[str, Path]:
    target_dir = output_dir or OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = target_dir / "monitoring-operations-brief.md"
    html_path = target_dir / "monitoring-operations-brief.html"
    markdown_path.write_text(build_markdown_report(csv_path), encoding="utf-8")
    html_path.write_text(build_html_report(csv_path), encoding="utf-8")
    return {"markdown": markdown_path, "html": html_path}


def main() -> None:
    print(build_markdown_report())
    output_paths = export_reports()
    print(f"\nWrote {output_paths['markdown']} and {output_paths['html']}")


if __name__ == "__main__":
    main()