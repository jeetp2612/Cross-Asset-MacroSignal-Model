import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from logic import (
    ACCENT,
    AMBER,
    ASSET_COLORS,
    ASSETS,
    BORDER,
    CARD_BG,
    DARK,
    DEFAULT_COMPARE,
    FOLDER,
    FULL,
    GREEN,
    GREY,
    INSIGHT_BG,
    MODEL_LABELS,
    PAGE_BG,
    PERIODS,
    PRESETS,
    RED,
    SHORT,
    VIEWS,
    __version__,
    alert_text,
    asset_table,
    backtest_rows,
    backtest_summary,
    backtest_text,
    chart_frame,
    curve_stats,
    dashboard_message,
    data_age_text,
    draw_backtest,
    draw_compare,
    draw_forecast,
    draw_markets,
    draw_reality,
    draw_whatif,
    forecast_summary,
    forecast_text,
    hover_values,
    label_color,
    link_of,
    load_backtest,
    load_file,
    load_forecast,
    load_market,
    market_facts,
    market_snapshot,
    markets_text,
    mix_forecast_vol,
    mood_of,
    nearest_row,
    normalise,
    parse_amount,
    period_slice,
    proof_of,
    reality_rows,
    reality_text,
    replay_history,
    risk_of_swing,
    risk_rows,
    scoreboard_rows,
    whatif_text,
)
from updater import Updater

matplotlib.use("TkAgg")

TAB_TITLES = ["Dashboard", "Markets", "Risk Forecast", "Build Your Mix", "Put to Work", "Reality Check"]

TITLE_FONT = ("Segoe UI", 15, "bold")
HEAD_FONT = ("Segoe UI", 11, "bold")
BODY_FONT = ("Segoe UI", 10)
SMALL_FONT = ("Segoe UI", 9)
BIG_FONT = ("Segoe UI", 22, "bold")

FRESH_ON_DARK = "#8fe3b0"
OLD_ON_DARK = "#ffd27f"

CARD_NOTES = {
    "fear": "Under 15 is calm, over 25 is fearful.",
    "curve": "Below zero (inverted) has often come before slowdowns.",
    "link": "Opposite moves cushion a portfolio.",
}

HOW_IT_WORKS = (
    "How this works:  1) Exploring the data showed that risk comes in waves, so recent risk helps predict next month's risk.  "
    "2) Models learn from recent risk, the fear index, credit and economic signals.  "
    "3) Every model is tested year by year, using only the past, and compared with the naive rule 'next month will look like this month'."
)


class Card:
    def __init__(self, parent, title, wrap):
        self.frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(self.frame, text=title, font=SMALL_FONT, bg=CARD_BG, fg=GREY, anchor="w").pack(fill="x", padx=12, pady=(10, 0))
        self.big = tk.Label(self.frame, text="", font=BIG_FONT, bg=CARD_BG, fg=DARK, anchor="w")
        self.big.pack(fill="x", padx=12)
        self.status = tk.Label(self.frame, text="", font=HEAD_FONT, bg=CARD_BG, fg=DARK, anchor="w")
        self.status.pack(fill="x", padx=12)
        self.note = tk.Label(self.frame, text="", font=SMALL_FONT, bg=CARD_BG, fg=GREY, wraplength=wrap, justify="left", anchor="w")
        self.note.pack(fill="x", padx=12, pady=(2, 10))

    def set(self, big, status, color, note):
        self.big.config(text=big, fg=color)
        self.status.config(text=status, fg=color)
        self.note.config(text=note)


def make_button(parent, text, command, primary=False):
    if primary:
        return tk.Button(parent, text=text, command=command, font=BODY_FONT, bg=ACCENT, fg="white", activebackground=DARK, activeforeground="white", relief="flat", padx=14, pady=5, cursor="hand2")
    return tk.Button(parent, text=text, command=command, font=BODY_FONT, bg="white", fg=DARK, activebackground=BORDER, relief="solid", bd=1, padx=10, pady=3, cursor="hand2")


def make_choice(parent, text, value, variable, command, width):
    return tk.Radiobutton(parent, text=text, value=value, variable=variable, command=command, indicatoron=False, width=width, font=BODY_FONT, bg="white", fg=DARK, selectcolor="#cfe0f3", activebackground=BORDER, relief="solid", bd=1, cursor="hand2")


def make_table(parent, columns, widths, height):
    table = ttk.Treeview(parent, columns=columns, show="headings", height=height, selectmode="none")
    for name, width in zip(columns, widths):
        table.heading(name, text=name)
        table.column(name, width=width, anchor="w" if name == columns[0] else "e", stretch=True)
    table.tag_configure("calm", foreground=GREEN)
    table.tag_configure("usual", foreground=DARK)
    table.tag_configure("wild", foreground=RED)
    return table


def clear_table(table):
    for item in table.get_children():
        table.delete(item)


def tag_of(label):
    if label is None:
        return "usual"
    if label.startswith("Calmer"):
        return "calm"
    if label.startswith("Wilder"):
        return "wild"
    return "usual"


class App:
    def __init__(self, root, data, summary, forecast=None, width=1100, backtest=None):
        self.root = root
        self.data = data
        self.summary = summary
        self.forecast = forecast
        self.backtest = backtest
        self.wrap = width - 80
        self.updater = None
        self.chart_data = None
        self.hover_line = None
        self.fact_widgets = []
        self.mix_label = None
        self.years = [str(year) for year in sorted(set(data.index.year))]

        root.configure(bg=PAGE_BG)
        self.setup_style()
        self.build_header()
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(8, 0))
        self.build_dashboard()
        self.build_markets()
        self.build_risk()
        self.build_mix()
        self.build_work()
        self.build_reality()
        tk.Label(
            root,
            text="Educational tool using real past prices. Not investment advice. Past results do not predict future returns.",
            font=SMALL_FONT,
            bg=PAGE_BG,
            fg=GREY,
        ).pack(fill="x", pady=4)

    def setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=PAGE_BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=HEAD_FONT, padding=(18, 8), background="#e3e8ef", foreground=GREY)
        style.map("TNotebook.Tab", background=[("selected", CARD_BG)], foreground=[("selected", DARK)])
        style.configure("Header.TButton", font=SMALL_FONT, padding=(8, 4), foreground="white", background=DARK, borderwidth=0)
        style.map(
            "Header.TButton",
            foreground=[("disabled", "#9fb3c8")],
            background=[("pressed", ACCENT), ("active", "#344e68")],
        )
        style.configure("HeaderPrimary.TButton", font=SMALL_FONT, padding=(10, 4), foreground="white", background=ACCENT, borderwidth=0)
        style.map(
            "HeaderPrimary.TButton",
            foreground=[("disabled", "#d9e2ec")],
            background=[("disabled", "#526b84"), ("pressed", DARK), ("active", "#4a86c8")],
        )
        style.configure("TCombobox", padding=4)
        style.configure("Treeview", font=BODY_FONT, rowheight=28, background=CARD_BG, fieldbackground=CARD_BG, bordercolor=BORDER)
        style.configure("Treeview.Heading", font=HEAD_FONT, background="#e8edf3", foreground=DARK)

    def build_header(self):
        top = tk.Frame(self.root, bg=DARK)
        top.pack(fill="x")
        tk.Label(top, text="Cross-Asset Macro Signal", font=TITLE_FONT, bg=DARK, fg="white").pack(side="left", padx=14, pady=10)
        tk.Label(top, text="v" + __version__, font=SMALL_FONT, bg=DARK, fg="#9fb3c8").pack(side="left")
        ttk.Button(top, text="About", command=self.show_about, style="Header.TButton", cursor="hand2").pack(side="right", padx=10)
        self.update_button = ttk.Button(top, text="Update data", command=self.start_update, style="HeaderPrimary.TButton", cursor="hand2")
        self.update_button.pack(side="right", padx=4)
        text, fresh = data_age_text(self.data)
        self.fresh_label = tk.Label(top, text=text, font=SMALL_FONT, bg=DARK, fg=FRESH_ON_DARK if fresh else OLD_ON_DARK)
        self.fresh_label.pack(side="right", padx=10)

    def add_page(self, title):
        page = tk.Frame(self.tabs, bg=PAGE_BG)
        self.tabs.add(page, text=title)
        return page

    def add_chart(self, parent, size):
        fig = Figure(figsize=size, dpi=100, layout="constrained", facecolor=CARD_BG)
        holder = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        holder.pack(fill="both", expand=True)
        canvas = FigureCanvasTkAgg(fig, master=holder)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, canvas

    def add_section(self, parent, title, hint):
        row = tk.Frame(parent, bg=PAGE_BG)
        row.pack(fill="x", padx=12, pady=(10, 2))
        tk.Label(row, text=title, font=HEAD_FONT, bg=PAGE_BG, fg=DARK).pack(side="left")
        tk.Label(row, text=hint, font=SMALL_FONT, bg=PAGE_BG, fg=GREY).pack(side="left", padx=10)
        return row

    def add_note(self, parent, text, background=INSIGHT_BG):
        label = tk.Label(parent, text=text, font=BODY_FONT, bg=background, fg=DARK, wraplength=self.wrap - 20, justify="left", anchor="w", padx=12, pady=8)
        label.pack(fill="x", padx=12, pady=4)
        return label

    def add_cards(self, parent, titles):
        row = tk.Frame(parent, bg=PAGE_BG)
        row.pack(fill="x", padx=8)
        cards = [Card(row, title, self.wrap // len(titles) - 40) for title in titles]
        for i, card in enumerate(cards):
            row.columnconfigure(i, weight=1, uniform="cards")
            card.frame.grid(row=0, column=i, sticky="nsew", padx=4, pady=2)
        return cards

    def show_about(self):
        messagebox.showinfo(
            "About",
            "Cross-Asset Macro Signal v" + __version__ + "\n\n"
            "Shows how stocks, bonds, gold and the US dollar behave, what the fear index and yield curve say right now, "
            "forecasts next month's risk, and tests honestly what can and cannot be predicted.\n\n"
            "Data: prices and VIX from Yahoo Finance (through the yfinance library), interest rates and economic series from FRED "
            "(St. Louis Fed). These sources are free for personal and educational use. Check their terms before any other use.\n\n"
            "This is an educational tool, not investment advice. Past results do not predict future returns.",
        )

    def start_update(self):
        if self.updater is not None:
            return
        question = "This downloads the latest data (internet needed) and re-runs the forecast tests.\nIt can take several minutes. Continue?"
        if not messagebox.askyesno("Update data", question):
            return
        self.update_button.config(state="disabled")
        self.updater = Updater(FOLDER)
        self.updater.start()
        self.poll_update()

    def poll_update(self):
        updater = self.updater
        if not updater.finished:
            self.fresh_label.config(text=updater.status(), fg="white")
            self.root.after(400, self.poll_update)
            return
        if updater.error:
            self.updater = None
            self.update_button.config(state="normal")
            text, fresh = data_age_text(self.data)
            self.fresh_label.config(text=text, fg=FRESH_ON_DARK if fresh else OLD_ON_DARK)
            messagebox.showerror("Update failed", updater.error + "\n\nYour old data was kept.")
            return
        messagebox.showinfo("Update finished", "The data is up to date. The app will now restart.")
        restart(self.root)

    def build_dashboard(self):
        page = self.add_page(TAB_TITLES[0])
        shot = market_snapshot(self.data)
        headline, text, color = dashboard_message(shot)

        banner = tk.Frame(page, bg=color)
        banner.pack(fill="x", padx=12, pady=(12, 6))
        inside = tk.Frame(banner, bg=CARD_BG)
        inside.pack(fill="x", padx=(6, 0))
        tk.Label(inside, text=headline, font=("Segoe UI", 14, "bold"), bg=CARD_BG, fg=color, anchor="w").pack(fill="x", padx=14, pady=(8, 0))
        tk.Label(inside, text=text + f"  (Data up to {shot['date']})", font=BODY_FONT, bg=CARD_BG, fg=GREY, wraplength=self.wrap, justify="left", anchor="w").pack(fill="x", padx=14, pady=(0, 8))

        cards = self.add_cards(page, ["Fear meter (VIX)", "Yield curve", "Stocks vs bonds", "Last 12 months", "Next-month risk (Stocks)"])
        mood, mood_color = mood_of(shot["vix"])
        cards[0].set(f"{shot['vix']:.1f}", mood, mood_color, f"Higher than {shot['rank']:.0f}% of days since {shot['first_year']}. " + CARD_NOTES["fear"])
        if shot["curve"] < 0:
            cards[1].set(f"{shot['curve']:+.2f}", "Inverted", RED, CARD_NOTES["curve"])
        else:
            cards[1].set(f"{shot['curve']:+.2f}", "Normal", GREEN, CARD_NOTES["curve"])
        link_word, link_color = link_of(shot["link"])
        cards[2].set(f"{shot['link']:+.2f}", link_word, link_color, "Last year of daily moves. " + CARD_NOTES["link"])
        changes = shot["changes"]
        best = shot["best"]
        year_line = ", ".join(f"{SHORT[a]} {changes[a]:+.0f}%" for a in ASSETS)
        cards[3].set(SHORT[best], "led the way", GREEN if changes[best] >= 0 else RED, year_line)
        if self.forecast is not None:
            info = forecast_summary(self.forecast, "SPY")
            cards[4].set(f"{info['forecast']:.0f}%", info["label"], label_color(info["label"]), f"Expected yearly swing. Usual range {info['low_cut']:.0f}% to {info['high_cut']:.0f}%.")
        else:
            cards[4].set("n/a", "Not built yet", GREY, "Press Update data to build the risk forecast.")

        self.add_section(page, "Compare the markets", "Tick markets, choose a view and a period, then move your mouse over the chart")
        controls = tk.Frame(page, bg=PAGE_BG)
        controls.pack(fill="x", padx=12)
        self.compare_vars = {}
        for code in ASSETS:
            variable = tk.BooleanVar(value=code in DEFAULT_COMPARE)
            self.compare_vars[code] = variable
            tk.Checkbutton(controls, text=SHORT[code], variable=variable, command=self.draw_dashboard_chart, font=HEAD_FONT, bg=PAGE_BG, fg=ASSET_COLORS[code], activebackground=PAGE_BG, activeforeground=ASSET_COLORS[code], selectcolor=CARD_BG, cursor="hand2").pack(side="left", padx=(0, 8))
        self.period_var = tk.StringVar(value="5Y")
        for label in reversed(list(PERIODS)):
            make_choice(controls, label, label, self.period_var, self.draw_dashboard_chart, 4).pack(side="right", padx=2)
        tk.Label(controls, text="Period", font=SMALL_FONT, bg=PAGE_BG, fg=GREY).pack(side="right", padx=6)
        self.view_var = tk.StringVar(value=VIEWS[0])
        for view in reversed(VIEWS):
            make_choice(controls, view, view, self.view_var, self.draw_dashboard_chart, 9).pack(side="right", padx=2)
        tk.Label(controls, text="View", font=SMALL_FONT, bg=PAGE_BG, fg=GREY).pack(side="right", padx=(14, 6))

        body = tk.Frame(page, bg=PAGE_BG)
        body.pack(fill="both", expand=True, padx=12, pady=(6, 4))
        self.facts_panel = tk.Frame(body, bg=PAGE_BG, width=260)
        self.facts_panel.pack(side="right", fill="y", padx=(10, 0))
        chart_side = tk.Frame(body, bg=PAGE_BG)
        chart_side.pack(side="left", fill="both", expand=True)
        self.dash_fig, self.dash_canvas = self.add_chart(chart_side, (8, 4))
        self.hover_label = tk.Label(chart_side, text="", font=BODY_FONT, bg=PAGE_BG, fg=DARK, anchor="w")
        self.hover_label.pack(fill="x", pady=(2, 0))
        self.dash_canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.draw_dashboard_chart()

    def draw_dashboard_chart(self):
        codes = [code for code in ASSETS if self.compare_vars[code].get()]
        label = self.period_var.get()
        view = self.view_var.get()
        frame = period_slice(self.data, label)
        self.chart_data = chart_frame(frame, codes, view) if codes else None
        ax = draw_compare(self.dash_fig, self.chart_data, codes, view)
        self.hover_line = ax.axvline(frame.index[0], color="#444444", linewidth=0.8, visible=False)
        self.hover_label.config(text="Move your mouse over the chart to see the values on any day." if codes else "")
        self.dash_canvas.draw()
        self.show_facts(frame, codes, label)

    def show_facts(self, frame, codes, label):
        for widget in self.fact_widgets:
            widget.destroy()
        self.fact_widgets = []
        period_name = "all the data" if label == "All" else "the last " + label.replace("Y", " year" + ("s" if label != "1Y" else ""))
        heading = tk.Label(self.facts_panel, text="Over " + period_name, font=HEAD_FONT, bg=PAGE_BG, fg=DARK, anchor="w")
        heading.pack(fill="x", pady=(0, 4))
        self.fact_widgets.append(heading)
        for code in codes:
            facts = market_facts(frame, code)
            risk, risk_color = risk_of_swing(facts["swing"])
            box = tk.Frame(self.facts_panel, bg=CARD_BG, highlightbackground=ASSET_COLORS[code], highlightthickness=2)
            box.pack(fill="x", pady=3)
            tk.Label(box, text=FULL[code], font=HEAD_FONT, bg=CARD_BG, fg=ASSET_COLORS[code], anchor="w").pack(fill="x", padx=10, pady=(6, 0))
            total_color = GREEN if facts["total"] >= 0 else RED
            tk.Label(box, text=f"Return: {facts['total']:+.0f}%", font=BODY_FONT, bg=CARD_BG, fg=total_color, anchor="w").pack(fill="x", padx=10)
            tk.Label(box, text=f"Worst fall from a peak: {facts['worst']:.0f}%", font=BODY_FONT, bg=CARD_BG, fg=GREY, anchor="w").pack(fill="x", padx=10)
            tk.Label(box, text=f"Risk level: {risk} ({facts['swing']:.0f}% a year)", font=HEAD_FONT, bg=CARD_BG, fg=risk_color, anchor="w").pack(fill="x", padx=10, pady=(0, 6))
            self.fact_widgets.append(box)
        if not codes:
            note = tk.Label(self.facts_panel, text="Tick a market above to see its numbers.", font=BODY_FONT, bg=PAGE_BG, fg=GREY, wraplength=240, justify="left", anchor="w")
            note.pack(fill="x")
            self.fact_widgets.append(note)

    def on_hover(self, event):
        if event.inaxes is None or event.xdata is None or self.chart_data is None or self.hover_line is None:
            return
        date, row = nearest_row(self.chart_data, event.xdata)
        self.hover_line.set_xdata([date, date])
        self.hover_line.set_visible(True)
        note = "  (100 = value at the start of the period)" if self.view_var.get() == "Growth" else ""
        self.hover_label.config(text=f"{date.date()}    {hover_values(row, self.view_var.get())}{note}")
        self.dash_canvas.draw_idle()

    def build_markets(self):
        page = self.add_page(TAB_TITLES[1])
        self.add_section(page, "All markets at a glance", "Changes, current risk and next month's expected risk")
        columns = ["Market", "Price", "1 month", "3 months", "12 months", "Risk now", "From 1-year peak", "Next-month risk"]
        widths = [150, 90, 90, 90, 90, 90, 130, 190]
        holder = tk.Frame(page, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        holder.pack(fill="x", padx=12, pady=4)
        self.market_table = make_table(holder, columns, widths, len(ASSETS))
        self.market_table.pack(fill="x")
        self.fill_market_table()

        self.add_section(page, "How the markets move together", "Choose the period for the charts")
        controls = tk.Frame(page, bg=PAGE_BG)
        controls.pack(fill="x", padx=12)
        self.markets_period = tk.StringVar(value="All")
        for label in PERIODS:
            make_choice(controls, label, label, self.markets_period, self.draw_markets_tab, 4).pack(side="left", padx=2)
        self.markets_note = self.add_note(page, "")
        holder = tk.Frame(page, bg=PAGE_BG)
        holder.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        self.markets_fig, self.markets_canvas = self.add_chart(holder, (10, 3.6))
        self.draw_markets_tab()

    def fill_market_table(self):
        clear_table(self.market_table)
        for row in asset_table(self.data, self.forecast):
            if row["forecast"] is None:
                forecast_text_value = "n/a"
            else:
                forecast_text_value = f"{row['forecast']:.1f}%  {row['label']}"
            values = (
                FULL[row["asset"]],
                f"{row['price']:.2f}",
                f"{row['change_1m']:+.1f}%",
                f"{row['change_3m']:+.1f}%",
                f"{row['change_12m']:+.1f}%",
                f"{row['vol_now']:.1f}%",
                f"{row['from_peak']:.1f}%",
                forecast_text_value,
            )
            self.market_table.insert("", "end", values=values, tags=(tag_of(row["label"]),))

    def draw_markets_tab(self):
        frame = period_slice(self.data, self.markets_period.get())
        self.markets_note.config(text=markets_text(frame))
        draw_markets(self.markets_fig, frame)
        self.markets_canvas.draw()

    def build_risk(self):
        page = self.add_page(TAB_TITLES[2])
        self.add_section(page, "Next month's risk forecast", "How wild will each market be over the next 21 trading days?")
        self.add_note(page, HOW_IT_WORKS)
        controls = tk.Frame(page, bg=PAGE_BG)
        controls.pack(fill="x", padx=12, pady=2)
        self.risk_asset = tk.StringVar(value="SPY")
        for code in ASSETS:
            make_choice(controls, SHORT[code], code, self.risk_asset, self.draw_risk_tab, 9).pack(side="left", padx=2)
        self.model_choices = {"Best model for this market": "best"}
        for code, label in MODEL_LABELS.items():
            self.model_choices[label] = code
        self.risk_model = tk.StringVar(value="Best model for this market")
        picker = ttk.Combobox(controls, textvariable=self.risk_model, values=list(self.model_choices), width=34, state="readonly")
        picker.pack(side="right")
        picker.bind("<<ComboboxSelected>>", lambda event: self.draw_risk_tab())
        tk.Label(controls, text="Model", font=SMALL_FONT, bg=PAGE_BG, fg=GREY).pack(side="right", padx=6)

        self.risk_cards = self.add_cards(page, ["Risk over the last month", "Forecast for next month", "Chance of a wild month", "How reliable is it?"])
        self.risk_note = self.add_note(page, "", INSIGHT_BG)
        body = tk.Frame(page, bg=PAGE_BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        side = tk.Frame(body, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        side.pack(side="right", fill="y", padx=(10, 0))
        tk.Label(side, text="Scoreboard (tested year by year)", font=HEAD_FONT, bg=CARD_BG, fg=DARK, anchor="w").pack(fill="x", padx=10, pady=(8, 4))
        columns = ["Model", "Error cut", "Years better", "Real or luck?"]
        self.score_table = make_table(side, columns, [200, 85, 105, 100], len(MODEL_LABELS))
        self.score_table.pack(padx=6, pady=(0, 8))
        tk.Label(side, text="Error cut: how much smaller the forecast error is than the naive rule.\nReal or luck?: statistical test (Diebold-Mariano) of whether the gain is more than chance.", font=SMALL_FONT, bg=CARD_BG, fg=GREY, justify="left", wraplength=440, anchor="w").pack(fill="x", padx=10, pady=(0, 8))
        chart_side = tk.Frame(body, bg=PAGE_BG)
        chart_side.pack(side="left", fill="both", expand=True)
        self.risk_fig, self.risk_canvas = self.add_chart(chart_side, (7, 3.4))
        self.draw_risk_tab()

    def draw_risk_tab(self):
        asset = self.risk_asset.get()
        model = self.model_choices[self.risk_model.get()]
        draw_forecast(self.risk_fig, self.forecast, asset, model)
        self.risk_canvas.draw()
        clear_table(self.score_table)
        if self.forecast is None:
            for card in self.risk_cards:
                card.set("n/a", "Not built yet", GREY, "Press Update data.")
            self.risk_note.config(text="The risk forecast has not been built yet. Press Update data (top right) to download the data and run the tests.")
            return
        info = forecast_summary(self.forecast, asset)
        self.risk_cards[0].set(f"{info['current']:.1f}%", "yearly swing", ACCENT, f"Measured up to {info['date']}")
        self.risk_cards[1].set(f"{info['forecast']:.1f}%", info["label"], label_color(info["label"]), f"Usual range {info['low_cut']:.1f}% to {info['high_cut']:.1f}%")
        if info["wild_prob"] is None:
            self.risk_cards[2].set(f"{info['rank']:.0f}%", "of past months were calmer", GREY, "A higher number means a wilder month than most.")
        else:
            alert_color = RED if info["wild_prob"] >= 50 else (AMBER if info["wild_prob"] >= 25 else GREEN)
            self.risk_cards[2].set(f"{info['wild_prob']:.0f}%", "chance of a wild month", alert_color, "Wild = top quarter of past risk. Usual chance is 25%.")
        improvement_color = GREEN if info["improvement"] > 0 else RED
        if info["p_value"] is None:
            proof_text = ""
        else:
            proof_text = f"Real or luck? {proof_of(info['p_value'])[0]} (p = {info['p_value']:.3f}). "
        self.risk_cards[3].set(f"{info['improvement']:+.0f}%", "error vs naive rule", improvement_color, proof_text + f"Better in {info['years_beating']} of {info['years_total']} years.")
        self.risk_note.config(text=forecast_text(self.forecast, asset) + "\n" + alert_text(self.forecast, asset))
        for row in scoreboard_rows(self.forecast, asset):
            self.score_table.insert("", "end", values=row, tags=("usual",))

    def build_mix(self):
        page = self.add_page(TAB_TITLES[3])
        self.add_section(page, "Build your own mix", "Choose how to split your money and see how it would have done, and how risky it looks now")
        self.whatif_year = tk.StringVar(value=self.years[0])
        self.whatif_amount = tk.StringVar(value="10000")
        self.weight_vars = [tk.DoubleVar(value=value) for value in PRESETS["Classic 60/40"]]

        body = tk.Frame(page, bg=PAGE_BG)
        body.pack(fill="both", expand=True, padx=12, pady=4)
        side = tk.Frame(body, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        side.pack(side="left", fill="y", padx=(0, 10))
        main = tk.Frame(body, bg=PAGE_BG)
        main.pack(side="left", fill="both", expand=True)

        tk.Label(side, text="1. Start year", font=HEAD_FONT, bg=CARD_BG, fg=DARK, anchor="w").pack(fill="x", padx=14, pady=(12, 2))
        year_box = ttk.Combobox(side, textvariable=self.whatif_year, values=self.years, width=8, state="readonly")
        year_box.pack(anchor="w", padx=14)
        year_box.bind("<<ComboboxSelected>>", lambda event: self.run_whatif())
        tk.Label(side, text="2. Money to invest ($)", font=HEAD_FONT, bg=CARD_BG, fg=DARK, anchor="w").pack(fill="x", padx=14, pady=(12, 2))
        ttk.Entry(side, textvariable=self.whatif_amount, width=12).pack(anchor="w", padx=14)
        tk.Label(side, text="3. Pick a quick mix", font=HEAD_FONT, bg=CARD_BG, fg=DARK, anchor="w").pack(fill="x", padx=14, pady=(12, 2))
        presets = tk.Frame(side, bg=CARD_BG)
        presets.pack(fill="x", padx=14)
        for i, name in enumerate(PRESETS):
            make_button(presets, name, lambda key=name: self.use_preset(key)).grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
        tk.Label(side, text="or fine-tune it", font=HEAD_FONT, bg=CARD_BG, fg=DARK, anchor="w").pack(fill="x", padx=14, pady=(12, 0))
        for i, code in enumerate(ASSETS):
            cell = tk.Frame(side, bg=CARD_BG)
            cell.pack(fill="x", padx=14)
            tk.Label(cell, text=SHORT[code], width=7, anchor="w", font=BODY_FONT, bg=CARD_BG, fg=ASSET_COLORS[code]).pack(side="left")
            scale = tk.Scale(cell, from_=0, to=100, resolution=5, orient="horizontal", length=170, variable=self.weight_vars[i], command=self.mix_changed, bg=CARD_BG, highlightthickness=0, troughcolor=BORDER, showvalue=False)
            scale.bind("<ButtonRelease-1>", lambda event: self.run_whatif())
            scale.pack(side="left")
        self.mix_label = tk.Label(side, text="", font=BODY_FONT, bg=CARD_BG, fg=DARK, wraplength=250, justify="left", anchor="w")
        self.mix_label.pack(fill="x", padx=14, pady=(8, 4))
        make_button(side, "Show result", self.run_whatif, primary=True).pack(fill="x", padx=14, pady=(4, 14))

        self.result_cards = self.add_cards(main, ["Your money would be", "Growth per year", "Worst fall from a peak", "Expected risk next month"])
        self.whatif_insight = tk.Label(main, text="", font=BODY_FONT, bg=INSIGHT_BG, fg=DARK, wraplength=self.wrap - 330, justify="left", anchor="w", padx=10, pady=6)
        self.whatif_insight.pack(fill="x", padx=4, pady=4)
        self.whatif_fig, self.whatif_canvas = self.add_chart(main, (7, 3.4))
        self.mix_changed()
        self.run_whatif()

    def use_preset(self, name):
        for variable, value in zip(self.weight_vars, PRESETS[name]):
            variable.set(value)
        self.mix_changed()
        self.run_whatif()

    def mix_changed(self, value=None):
        if self.mix_label is None:
            return
        values = [variable.get() for variable in self.weight_vars]
        try:
            shares = normalise(values)
        except ValueError as error:
            self.mix_label.config(text=str(error))
            return
        parts = [f"{SHORT[code]} {share * 100:.0f}%" for code, share in zip(ASSETS, shares) if share > 0]
        self.mix_label.config(text="Your mix: " + ", ".join(parts))

    def run_whatif(self):
        try:
            amount = parse_amount(self.whatif_amount.get())
            weights = normalise([variable.get() for variable in self.weight_vars])
            mix_curve, stock_curve = replay_history(self.data, int(self.whatif_year.get()), weights)
        except ValueError as error:
            messagebox.showwarning("Check your choices", str(error))
            return
        mix = curve_stats(mix_curve)
        stocks = curve_stats(stock_curve)
        risk, risk_color = risk_of_swing(mix["swing"])
        total_color = GREEN if mix["total"] >= 0 else RED
        self.result_cards[0].set(f"${mix['final'] * amount:,.0f}", f"{mix['total']:+.0f}% in total", total_color, f"100% stocks: ${stocks['final'] * amount:,.0f}")
        self.result_cards[1].set(f"{mix['yearly']:.1f}%", "average per year", ACCENT, f"100% stocks: {stocks['yearly']:.1f}%")
        self.result_cards[2].set(f"{mix['worst']:.0f}%", f"{risk} risk", risk_color, f"100% stocks: {stocks['worst']:.0f}%")
        expected = mix_forecast_vol(self.data, weights, self.forecast)
        if expected is None:
            self.result_cards[3].set("n/a", "Not built yet", GREY, "Press Update data to build the risk forecast.")
        else:
            stock_expected = mix_forecast_vol(self.data, [1, 0, 0, 0], self.forecast)
            expected_risk, expected_color = risk_of_swing(expected)
            self.result_cards[3].set(f"{expected:.1f}%", f"{expected_risk} risk", expected_color, f"Yearly swing. 100% stocks: {stock_expected:.1f}%")
        self.whatif_insight.config(text=whatif_text(mix_curve, stock_curve, weights))
        draw_whatif(self.whatif_fig, mix_curve, stock_curve, amount, self.whatif_year.get())
        self.whatif_canvas.draw()

    def build_work(self):
        page = self.add_page(TAB_TITLES[4])
        self.add_section(page, "Put the forecast to work", "Hold less when the forecast says risk is high. Is that worth it in money terms?")
        self.add_note(page, "The rule: each day, position = usual risk divided by forecast risk, never above 100% (no borrowing), with a 0.05% cost for every change. Money not invested sits in cash earning nothing.")
        controls = tk.Frame(page, bg=PAGE_BG)
        controls.pack(fill="x", padx=12, pady=2)
        self.work_asset = tk.StringVar(value="SPY")
        for code in ASSETS:
            make_choice(controls, SHORT[code], code, self.work_asset, self.draw_work_tab, 9).pack(side="left", padx=2)
        self.work_cards = self.add_cards(page, ["Yearly return", "Worst fall", "Return per unit of risk", "Average money invested"])
        self.work_note = self.add_note(page, "", INSIGHT_BG)
        body = tk.Frame(page, bg=PAGE_BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        side = tk.Frame(body, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        side.pack(side="right", fill="y", padx=(10, 0))
        tk.Label(side, text="All markets: buy and hold > forecast rule", font=HEAD_FONT, bg=CARD_BG, fg=DARK, anchor="w").pack(fill="x", padx=10, pady=(8, 4))
        columns = ["Market", "Yearly return", "Worst fall", "Sharpe"]
        self.work_table = make_table(side, columns, [75, 130, 110, 110], 4)
        self.work_table.pack(padx=6, pady=(0, 8))
        chart_side = tk.Frame(body, bg=PAGE_BG)
        chart_side.pack(side="left", fill="both", expand=True)
        self.work_fig, self.work_canvas = self.add_chart(chart_side, (7, 3.4))
        self.draw_work_tab()

    def draw_work_tab(self):
        asset = self.work_asset.get()
        draw_backtest(self.work_fig, self.backtest, asset)
        self.work_canvas.draw()
        clear_table(self.work_table)
        if self.backtest is None:
            for card in self.work_cards:
                card.set("n/a", "Not built yet", GREY, "Press Update data.")
            self.work_note.config(text="The risk-managed test has not been built yet. Press Update data (top right) to run it.")
            return
        info = backtest_summary(self.backtest, asset)
        used = info["used"]
        hold = info["hold"]
        self.work_cards[0].set(f"{used['yearly_return']:+.1f}%", "with the forecast rule", GREEN if info["return_gap"] >= -0.1 else AMBER, f"Buy and hold: {hold['yearly_return']:+.1f}% a year")
        self.work_cards[1].set(f"{used['worst_fall']:.0f}%", "worst fall from a peak", GREEN if info["fall_saved"] > 0.5 else GREY, f"Buy and hold: {hold['worst_fall']:.0f}%")
        self.work_cards[2].set(f"{used['sharpe']:.2f}", "Sharpe ratio", GREEN if info["sharpe_gap"] > 0.02 else (RED if info["sharpe_gap"] < -0.02 else GREY), f"Buy and hold: {hold['sharpe']:.2f}. Higher is better.")
        self.work_cards[3].set(f"{used['average_exposure']:.0f}%", "of your money invested", ACCENT, "The rest waited in cash on risky days.")
        self.work_note.config(text=backtest_text(self.backtest, asset))
        for row in backtest_rows(self.backtest):
            self.work_table.insert("", "end", values=row, tags=("usual",))

    def build_reality(self):
        page = self.add_page(TAB_TITLES[5])
        self.add_section(page, "What can be predicted, and what cannot?", "Every model was trained only on the past and tested year by year")
        risk = risk_rows(self.forecast)
        direction = reality_rows(self.summary)
        self.add_note(page, reality_text(risk, direction))
        holder = tk.Frame(page, bg=PAGE_BG)
        holder.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        self.reality_fig, self.reality_canvas = self.add_chart(holder, (10, 4))
        draw_reality(self.reality_fig, risk, direction)
        self.reality_canvas.draw()


def restart(root):
    subprocess.Popen([sys.executable, os.path.abspath(sys.argv[0])])
    root.destroy()


def open_app(root, width):
    data = load_market()
    if data is None:
        messagebox.showerror("Data not found", "The data files could not be created. Check your internet connection and start the app again.")
        root.destroy()
        return
    summary = load_file("walk_forward_summary.csv", False)
    forecast = load_forecast()
    App(root, data, summary, forecast, width, load_backtest())


def first_run(root, width):
    question = (
        "No data was found yet.\n\nDownload the latest data and set the app up now?\n"
        "This needs internet and takes several minutes, but only the first time."
    )
    if not messagebox.askyesno("First start", question):
        root.destroy()
        return False
    label = tk.Label(root, text="Starting...", font=HEAD_FONT, fg=DARK)
    label.pack(expand=True)
    updater = Updater(FOLDER)
    updater.start()

    def poll():
        if not updater.finished:
            label.config(text=updater.status() + "\nPlease keep this window open.")
            root.after(400, poll)
            return
        label.destroy()
        if updater.error:
            messagebox.showerror("Setup failed", updater.error + "\n\nCheck your internet connection and start the app again.")
            root.destroy()
            return
        open_app(root, width)

    root.after(400, poll)
    return True


def main():
    root = tk.Tk()
    root.title("Cross-Asset Macro Signal")
    width = min(1300, root.winfo_screenwidth() - 80)
    height = min(920, root.winfo_screenheight() - 120)
    root.geometry(f"{width}x{height}")
    root.minsize(1000, 680)
    if load_market() is None:
        if not first_run(root, width):
            return
    else:
        open_app(root, width)
    root.mainloop()


if __name__ == "__main__":
    main()
