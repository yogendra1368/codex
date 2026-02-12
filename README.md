# TM83 Grafana Operations Dashboard (Loki)

This repository now contains a production-style Grafana dashboard JSON for your setup:

- Loki datasource UID: `loki`
- Jobs: `provtrace`, `syserr`
- App label: `tm83`

## Import dashboard

1. Open Grafana → **Dashboards** → **Import**.
2. Upload `grafana-ops-loki-dashboard.json`.
3. Choose datasource **loki** when prompted.
4. Save.

## What this dashboard gives Operations

- **Top health KPIs**
  - Total error count in selected range.
  - Event loop activity gauge parsed from provtrace lines.
- **Traffic & trend view**
  - Log throughput by job over time.
- **Quality view**
  - Severity split pie chart from syserr logs.
- **Investigation views**
  - Parsed table with host/component/file/line/level/message.
  - Full logs explorer with free text filter.

## Expected labels

To get full value from filters, ensure Promtail pushes labels like:

- `job` (`provtrace` / `syserr`)
- `app` (`tm83`)
- `host` (node/hostname)

## Notes on parsing

The dashboard uses Loki `pattern` parser:

- provtrace pattern for event loop %:
  - `<ts> <dow> EventLoop Event loop activity <activity>%`
- syserr pattern:
  - `<mon> <day> <ts> <host> <pid> (<component>) <tag> <file> <line> <level> <msg>`

If log format varies slightly, update these pattern strings in the relevant panel query.
