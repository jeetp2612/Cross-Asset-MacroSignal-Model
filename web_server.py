import csv
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_ROOT = os.path.join(ROOT, "web")
DATA_ROOT = os.path.join(ROOT, "data")


def read_csv(name):
    path = os.path.join(DATA_ROOT, name)
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def number(value, digits=2):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_ROOT, **kwargs)

    def send_json(self, value):
        body = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/overview":
            latest = read_csv("forecast_latest.csv")
            scores = read_csv("forecast_scores.csv")
            market = read_csv("market.csv")
            latest_row = market[-1] if market else {}
            cards = []
            for row in latest:
                cards.append({
                    "asset": row["asset"],
                    "date": row["date"],
                    "current": number(row["current_vol"]),
                    "forecast": number(row["forecast_vol"]),
                    "model": row["best_model"],
                    "label": row["label"],
                    "wildProb": number(row["wild_prob"], 1),
                    "naive": number(row["naive"]),
                    "low": number(row["low_cut"]),
                    "high": number(row["high_cut"]),
                })
            score_rows = [{
                "asset": row["asset"],
                "model": row["model"],
                "rmse": number(row["rmse"]),
                "improvement": number(row["improvement_pct"]),
                "years": row["years_beating_naive"] + "/" + row["years_total"],
            } for row in scores]
            self.send_json({
                "cards": cards,
                "scores": score_rows,
                "marketDate": latest_row.get("date", ""),
                "vix": number(latest_row.get("VIX")),
                "curve": number(latest_row.get("CURVE_10Y_2Y")),
                "sp500": number(latest_row.get("SPY")),
            })
            return

        if path == "/api/history":
            rows = read_csv("forecast_predictions.csv")[-180:]
            history = [{
                "date": row["date"],
                "SPY": number(row["SPY_actual"]),
                "TLT": number(row["TLT_actual"]),
                "GLD": number(row["GLD_actual"]),
                "UUP": number(row["UUP_actual"]),
            } for row in rows]
            self.send_json(history)
            return

        if path == "/api/direction":
            rows = read_csv("walk_forward_summary.csv")
            self.send_json([{
                "asset": row["asset"],
                "model": row["model"],
                "accuracy": number(row["avg_auc"], 3),
                "strategy": number(row["strategy_return"], 3),
                "buyHold": number(row["buy_hold_return"], 3),
            } for row in rows])
            return

        super().do_GET()


if __name__ == "__main__":
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Open: http://127.0.0.1:8000")
    server.serve_forever()
