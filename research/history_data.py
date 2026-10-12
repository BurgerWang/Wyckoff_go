"""Long real-history acquisition, without imputation or rule-generated labels.

End is exclusive. Corporate-action adjusted Yahoo data are today's vintage,
not point-in-time prices. Binance history stays on the exact spot USDT pair.
"""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from market_data import normalize_download, read_csv, validate_ohlcv
from research.universe import PROFILES, Instrument, verify_identity
from research import binance_data as bd

# Only documented IPO/security boundaries may remove expected leading sessions.
BOUNDARIES = {
    "INTC": ("1971-10-13", "Intel IPO; common stock INTC, CUSIP 458140100; 1989 reincorporation predates requested window", ["https://www.intc.com/stock-info/faq"]),
    "ADBE": ("1986-08-20", "Adobe IPO; 1997 Delaware reincorporation predates requested window; Adobe Systems Incorporated renamed Adobe Inc. in 2018", ["https://www.adobe.com/investor-relations/faq.html"]),
    "META": ("2012-05-18", "Facebook Class A IPO; FB renamed META 2022-06-09, unchanged CUSIP", [
        "https://s21.q4cdn.com/399680738/files/doc_news/2012/FB_News_2012_5_17_Financial_Releases.pdf",
        "https://investor.atmeta.com/investor-news/press-release-details/2022/Meta-Platforms-Inc.-to-Change-Ticker-Symbol-to-META-on-June-9/default.aspx"]),
    "TSLA": ("2010-06-29", "Tesla IPO", ["https://ir.tesla.com/press-release/tesla-announces-pricing-initial-public-offering"]),
    "AVGO": ("2009-08-06", "Avago AVGO IPO; 2016 Broadcom successor shares exchanged 1:1; 2018 US redomiciliation. No BRCM history appended", [
        "https://investors.broadcom.com/news-releases/news-release-details/avago-technologies-limited-prices-initial-public-offering",
        "https://www.sec.gov/Archives/edgar/data/1649338/000164933816000151/avgo-10302016x10k.htm",
        "https://investors.broadcom.com/restricted-partnership-units"]),
    "PLTR": ("2020-09-30", "Palantir Class A direct listing", ["https://www.sec.gov/Archives/edgar/data/1321655/000119312520254706/d38725dex991.htm"]),
    "DELL": ("2018-12-28", "Dell Technologies Class C relisting; excludes old Dell Inc. pre-privatization and DVMT tracking shares", ["https://investors.delltechnologies.com/stock-information"]),
    "VRT": ("2020-02-10", "Vertiv first NYSE trading after 2020-02-07 business combination; excludes GSAH SPAC", ["https://investors.vertiv.com/financials/annual-reports/2021-Shareholder-Letter/"]),
    "SMCI": ("2007-03-29", "IPO; same common stock traded OTC during Nasdaq suspension/delisting 2018-08-23 to 2020-01-13; relisted 2020-01-14", [
        "https://www.supermicro.com/en/about",
        "https://www.sec.gov/Archives/edgar/data/1375365/000137536520000064/smci-2020630x10k.htm"]),
    "ANET": ("2014-06-06", "Arista IPO", ["https://www.arista.com/en/company/news/press-release/940-pr-20140606-01"]),
    "SPCX": ("2026-06-12", "SpaceX Class A only; excludes reused ticker's former ETF", ["https://m.nasdaqtrader.com/TraderNews.aspx?id=DTN2026-8"]),
}
CONTINUITY = {
    "IBM": {"note": "IBM common stock predates requested 2006 window; issuer annual report records share-price history since November 1915. 2021 Kyndryl spin-off is a corporate action, not a substituted ticker series; Yahoo current-vintage adjustments apply.", "sources": ["https://www.ibm.com/investor/help/general-faqs", "https://www.ibm.com/investor/att/pdf/IBM_Annual_Report_2010.pdf", "https://newsroom.ibm.com/2021-11-03-IBM-Completes-the-Separation-of-Kyndryl"]},
    "GOOGL": {"note": "Google Class A GOOG renamed GOOGL 2014-04-03; 2015 Alphabet successor class A shares exchanged 1:1. Not the separate GOOG Class C series.", "sources": [
        "https://www.sec.gov/Archives/edgar/data/1652044/000165204417000008/goog10-kq42016.htm",
        "https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2015-156"]},
}
PAIR_BOUNDARIES = {
    "INTC": ("1971-10-13", "Intel IPO; common stock INTC, CUSIP 458140100; 1989 reincorporation predates requested window", ["https://www.intc.com/stock-info/faq"]),
    "ADBE": ("1986-08-20", "Adobe IPO; 1997 Delaware reincorporation predates requested window; Adobe Systems Incorporated renamed Adobe Inc. in 2018", ["https://www.adobe.com/investor-relations/faq.html"]),
    "BTCUSDT": ("2017-08-17", "https://www.binance.com/zh-CN/support/announcement/detail/115001094331"),
    "ETHUSDT": ("2017-08-17", "https://www.binance.com/zh-CN/support/announcement/detail/115001094331"),
    "DOGEUSDT": ("2019-07-05", "https://www.binance.com/en-IN/support/announcement/detail/360030488211"),
}
LIMITATIONS = [
    "Yahoo auto_adjust=True is download-time adjusted OHLCV, not a point-in-time corporate-action archive.",
    "Current fixed equity pool has selection/survivorship bias; not the historical market universe.",
    "USD/share-volume equities and USDT/base-asset-volume spot pairs are separate markets.",
    "No imputation, proxy securities, cross-exchange crypto extension, bad-bar deletion, or fake pre-listing history.",
    "Missing sessions are disclosed individually and divide valid continuous segments; calendar expectations alone do not prove a halt.",
    "Yahoo OHLC adjustment float discrepancies <=8 ULP are enclosed and individually recorded; raw provider download is retained for new fetches.",
    "REST historical shortened intervals are replaced only by an independently validated exact-day official Binance archive bar; source rows and differences are retained.",
    "First Binance listing-day daily bar covers only post-launch trading within its UTC day; the completed bar is retained and disclosed.",
]


def window(start, end):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if a.tz is not None or b.tz is not None or a != a.normalize() or b != b.normalize() or a >= b:
        raise ValueError("start/end must be naive dates, start < end; end exclusive")
    if b > pd.Timestamp.now(tz="UTC").tz_localize(None).normalize():
        raise ValueError("end includes an unfinished UTC day")
    return a, b


def equity_start(symbol, start):
    return max(pd.Timestamp(start), pd.Timestamp(BOUNDARIES[symbol][0])) if symbol in BOUNDARIES else pd.Timestamp(start)


def normalize_adjustment_roundoff(raw, symbol, original=None):
    """Enclose only IEEE multiplication noise (<=8 ULP); retain raw download.

    Yahoo auto-adjust multiplies OHL but uses Adj Close directly. When raw
    close equals high/low, those equivalent arithmetic paths differ by ULPs.
    Genuine price violations still fail validation; no date/bar is removed.
    """
    f = raw.xs(symbol, axis=1, level=1) if isinstance(raw.columns, pd.MultiIndex) else raw
    fixes = []
    for side, bound in (("High", f[["Open", "High", "Low", "Close"]].max(axis=1)),
                        ("Low", f[["Open", "High", "Low", "Close"]].min(axis=1))):
        violation = f[side] < bound if side == "High" else f[side] > bound
        for date in f.index[violation]:
            before, after = float(f.loc[date, side]), float(bound.loc[date])
            endpoints = [c for c in ("Open", "Close") if float(f.loc[date, c]) == after]
            source = original.xs(symbol, axis=1, level=1) if original is not None and isinstance(original.columns, pd.MultiIndex) else original
            equal_source = next((c for c in endpoints if source is not None and date in source.index and
                                 float(source.loc[date, c]) == float(source.loc[date, side])), None)
            if equal_source and abs(after - before) <= 8 * abs(np.spacing(before)):
                column = (side, symbol) if isinstance(raw.columns, pd.MultiIndex) else side
                raw.loc[date, column] = after
                fixes.append({"date": date.strftime("%Y-%m-%d"), "field": side, "before": before, "after": after,
                              "reason": "auto-adjust equivalent arithmetic <=8 IEEE ULP", "absolute_delta": abs(after - before),
                              "original_equal_endpoint": equal_source, "original_price": float(source.loc[date, side])})
    return raw, fixes


def fetch_equity_history(instrument, start, end, *, raw_path=None):
    import yfinance as yf
    a, b = window(start, end)
    effective = equity_start(instrument.symbol, a)
    if effective >= b:
        raise ValueError("requested interval precedes documented security boundary")
    identity = verify_identity(instrument, yf.Ticker(instrument.symbol).get_info()) if instrument.identity_tokens else {"status": "not_required"}
    raw = yf.download(instrument.symbol, start=effective.strftime("%Y-%m-%d"), end=b.strftime("%Y-%m-%d"),
                      interval="1d", auto_adjust=True, progress=False, threads=False, timeout=30)
    if raw_path is not None:
        raw.to_csv(raw_path, index_label="Date")
    f = raw.xs(instrument.symbol, axis=1, level=1) if isinstance(raw.columns, pd.MultiIndex) else raw
    original = None
    if not f.empty and ((f.High < f[["Open", "Close"]].max(axis=1)) | (f.Low > f[["Open", "Close"]].min(axis=1))).any():
        original = yf.download(instrument.symbol, start=effective.strftime("%Y-%m-%d"), end=b.strftime("%Y-%m-%d"),
                               interval="1d", auto_adjust=False, progress=False, threads=False, timeout=30)
        if raw_path is not None:
            original.to_csv(raw_path.with_name(f"{instrument.symbol}.unadjusted-raw.csv"), index_label="Date")
    raw, numeric_fixes = normalize_adjustment_roundoff(raw, instrument.symbol, original)
    frame = normalize_download(raw, instrument.symbol)
    frame.attrs["adjustment_float_roundoff"] = numeric_fixes
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.attrs.update(timeframe="1d", market_type="cash", price_currency="USD", volume_unit="shares",
                       session_timezone="America/New_York", identity=identity,
                       source_urls=[f"https://query1.finance.yahoo.com/v8/finance/chart/{instrument.symbol}"],
                       fetched_at=pd.Timestamp.now(tz="UTC").isoformat(), auto_adjust=True,
                       point_in_time=False, adjustment_vintage="download_time")
    return frame


def first_pair_kline(symbol):
    """Query exact pair's earliest REST daily candle; official archive fallback."""
    listed, announcement = PAIR_BOUNDARIES[symbol]
    url = bd.REST_HOST + "/api/v3/klines?" + urlencode({"symbol": symbol, "interval": "1d", "startTime": 0, "limit": 1})
    urls = [url]
    transport = "rest_first_kline"
    try:
        payload = json.loads(bd._download(url))
        if not isinstance(payload, list) or len(payload) != 1:
            raise ValueError("first-kline response must contain exactly one candle")
        first = bd._parse_row(payload[0])[0]
    except HTTPError as exc:
        if exc.code not in (403, 451):
            raise
        transport = "official_archive_first_month_REST_restricted"
        # IPO date is independently documented; the monthly archive's earliest
        # candle confirms it. Do not treat an arbitrary downloaded start as IPO.
        rows = bd._archive_file(symbol, "monthly", listed[:7], urls)
        if not rows:
            raise ValueError("empty listing-month archive")
        first = min(t for t, _ in rows)
    if first.strftime("%Y-%m-%d") != listed:
        raise ValueError(f"{symbol}: first kline disagrees with official listing: {first} != {listed}")
    return {"date": listed, "announcement": announcement, "transport": transport, "source_urls": urls,
            "listing_day_partial_session": True}


def resolve_kline(row, symbol, archives, repairs, urls):
    try:
        return bd._parse_row(row)
    except ValueError as exc:
        if "close time is not a complete" not in str(exc):
            raise
        day = bd._timestamp(row[0])
        month = day.strftime("%Y-%m")
        if month not in archives:
            archives[month] = bd._archive_file(symbol, "monthly", month, urls)
        matched = [item for item in archives[month] if item[0] == day]
        if len(matched) != 1:
            raise ValueError(f"official archive cannot resolve incomplete REST interval {day}") from exc
        repaired = matched[0]
        repairs.append({"date": day.strftime("%Y-%m-%d"), "reason": str(exc), "rest_raw_row": row,
                        "replacement_source": "official_binance_monthly_archive",
                        "replacement_ohlcv_quotevolume": repaired[1],
                        "ohlcv_equal": [float(row[i]) for i in (1, 2, 3, 4, 5, 7)] == repaired[1],
                        "maintenance_context_source": "https://www.binance.com/en/support/announcement/detail/360000713851" if month == "2018-02" else None})
        return repaired


def history_rest(symbol, start, end, urls, repairs, raw_path=None):
    rows, raw_rows, archives = [], [], {}
    cursor = start
    while cursor < end:
        url = bd.REST_HOST + "/api/v3/klines?" + urlencode({"symbol": symbol, "interval": "1d", "timeZone": "0", "limit": 1000,
                                                               "startTime": cursor.value // 1_000_000, "endTime": end.value // 1_000_000 - 1})
        page = json.loads(bd._download(url))
        urls.append(url)
        if not isinstance(page, list) or len(page) > 1000:
            raise ValueError("invalid Binance kline response")
        raw_rows.extend(page)
        if raw_path is not None:
            _write(raw_path, {"source_urls": urls, "rows": raw_rows})
        if not page:
            break
        parsed = [resolve_kline(row, symbol, archives, repairs, urls) for row in page]
        times = [t for t, _ in parsed]
        if any(t < cursor or t >= end for t in times) or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("duplicate/unordered/out-of-range Binance UTC days")
        rows.extend(parsed)
        cursor = times[-1] + bd.DAY
    return rows


def fetch_pair_history(symbol, start, end, *, raw_path=None):
    a, b = window(start, end)
    first = first_pair_kline(symbol)
    beginning = max(a.tz_localize("UTC"), pd.Timestamp(first["date"], tz="UTC"))
    finish = b.tz_localize("UTC")
    if beginning >= finish:
        raise ValueError("requested interval precedes documented pair boundary")
    urls, repairs = [], []
    try:
        rows = history_rest(symbol, beginning, finish, urls, repairs, raw_path)
        transport, reason = "rest", None
    except HTTPError as exc:
        if exc.code not in (403, 451):
            raise
        urls = []
        rows = bd._archives(symbol, beginning, finish, urls)
        transport, reason = "binance_vision_archive", f"REST HTTP {exc.code}"
    frame = pd.DataFrame([values for _, values in rows],
                         index=pd.DatetimeIndex([t for t, _ in rows]).tz_localize(None), columns=bd.COLUMNS, dtype=float)
    frame.index.name = "Date"
    frame = validate_ohlcv(frame)
    if not np.isfinite(frame["QuoteVolume"].to_numpy()).all() or not (frame["QuoteVolume"] >= 0).all():
        raise ValueError("invalid quote volume")
    frame.attrs.update(symbol=symbol, source="binance_spot", timeframe="1d", market_type="spot",
                       price_currency="USDT", volume_unit=symbol[:-4], quote_volume_unit="USDT", adjusted=False,
                       transport=transport, fallback_reason=reason, first_kline=first, source_urls=urls,
                       official_archive_repairs=repairs,
                       fetched_at=pd.Timestamp.now(tz="UTC").isoformat(),
                       shortened_rest_intervals=[r["date"] for r in repairs],
                       trading_continuity="Exchange maintenance/outages may shorten actual trading; daily archive interval bounds do not prove uninterrupted trading")
    return frame


def coverage(frame, profile, instrument, start, end):
    frame = validate_ohlcv(frame)
    a, b = window(start, end)
    expected_attrs = {"symbol": instrument.symbol, "source": "yahoo" if profile.asset_class == "equity" else "binance_spot",
                      "timeframe": "1d", "market_type": profile.market_type, "price_currency": profile.price_currency,
                      "volume_unit": "shares" if profile.asset_class == "equity" else instrument.symbol[:-4],
                      "adjusted": profile.asset_class == "equity"}
    for k, v in expected_attrs.items():
        if frame.attrs.get(k) != v:
            raise ValueError(f"{instrument.symbol}: metadata mismatch: {k}")
    if frame.index.tz is not None or not frame.index.equals(frame.index.normalize()):
        raise ValueError("daily dates must be naive session/UTC midnights")
    if profile.asset_class == "equity":
        for fix in frame.attrs.get("adjustment_float_roundoff", []):
            if not fix.get("original_equal_endpoint") or abs(fix["after"] - fix["before"]) > 8 * abs(np.spacing(fix["before"])):
                raise ValueError("cached roundoff correction lacks raw equality proof or exceeds 8 ULP")
        import exchange_calendars as xcals
        a = equity_start(instrument.symbol, a)
        calendar = xcals.get_calendar("XNYS", start=a - pd.Timedelta(days=31), end=b + pd.Timedelta(days=31))
        expected = calendar.sessions_in_range(a, b - pd.Timedelta(days=1)).tz_localize(None)
        calendar_note = "XNYS exchange_calendars common US cash sessions; SMCI OTC period separately disclosed"
        if instrument.identity_tokens:
            identity = frame.attrs.get("identity", {})
            verify_identity(instrument, {"symbol": identity.get("symbol"), "longName": identity.get("name"),
                                        "quoteType": identity.get("quote_type"), "cusip": identity.get("provider_cusip")})
    else:
        if frame.attrs.get("quote_volume_unit") != "USDT" or "QuoteVolume" not in frame or not np.isfinite(frame["QuoteVolume"].to_numpy()).all() or (frame["QuoteVolume"] < 0).any():
            raise ValueError("Binance requires finite nonnegative USDT quote volume")
        listed = PAIR_BOUNDARIES[instrument.symbol][0]
        first = frame.attrs.get("first_kline", {})
        if first.get("date") != listed or first.get("announcement") != PAIR_BOUNDARIES[instrument.symbol][1]:
            raise ValueError("pair first-kline verification missing or mismatched")
        a = max(a, pd.Timestamp(listed))
        expected = pd.date_range(a, b, inclusive="left", freq="D")
        calendar_note = "UTC completed days; first listing day is a partial trading session"
    if (frame.index < a).any() or (frame.index >= b).any():
        raise ValueError("history outside documented security/request bounds")
    extra = frame.index.difference(expected)
    if len(extra):
        raise ValueError(f"unexpected dates: {extra.strftime('%Y-%m-%d').tolist()}")
    missing = expected.difference(frame.index)
    # Break segments at every missing expected session. Never delete gaps or
    # silently redefine the effective listing boundary to the provider's start.
    positions = expected.get_indexer(frame.index)
    splits = [0] + [i for i in range(1, len(positions)) if positions[i] != positions[i - 1] + 1] + [len(frame)]
    segments = [{"start": frame.index[l].strftime("%Y-%m-%d"), "end": frame.index[r - 1].strftime("%Y-%m-%d"), "bars": r - l}
                for l, r in zip(splits, splits[1:])]
    years = (frame.index[-1] - frame.index[0]).days / 365.25
    return {"effective_start": a.strftime("%Y-%m-%d"), "first": frame.index[0].strftime("%Y-%m-%d"),
            "last": frame.index[-1].strftime("%Y-%m-%d"), "bars": len(frame), "span_years": round(years, 3),
            "at_least_10_years": years >= 10, "expected_sessions": len(expected),
            "missing_dates": missing.strftime("%Y-%m-%d").tolist(), "missing_count": len(missing),
            "continuous_segments": segments, "calendar": calendar_note,
            "source_anomaly_dates": [r["date"] for r in frame.attrs.get("official_archive_repairs", [])],
            "zero_volume_dates": frame.index[frame.Volume.eq(0)].strftime("%Y-%m-%d").tolist(),
            "status": "complete" if not len(missing) else "partial_missing_sessions"}


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_history(output=Path("output/long-history/20260930"), start="2006-09-30", end="2026-09-30", *, refresh=False):
    """Download/checkpoint all 22 fixed instruments and SPY; no core replay."""
    window(start, end)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    request = {"start": str(start), "end": str(end), "end_exclusive": True}
    manifest = {"schema_version": 1, "request": request, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "limitations": LIMITATIONS, "profiles": {}}
    for profile in PROFILES.values():
        directory = output / profile.id / "data"
        directory.mkdir(parents=True, exist_ok=True)
        instruments = list(profile.instruments)
        if profile.benchmark not in [i.symbol for i in instruments]:
            instruments.append(Instrument(profile.benchmark, "Market benchmark", "基准"))
        entries = {}
        manifest["profiles"][profile.id] = entries
        for instrument in instruments:
            symbol = instrument.symbol
            csv, sidecar = directory / f"{symbol}.csv", directory / f"{symbol}.json"
            try:
                if not refresh and csv.exists() and sidecar.exists():
                    saved = json.loads(sidecar.read_text(encoding="utf-8"))
                    if saved.get("request") != request:
                        raise ValueError("cached history request differs; use refresh or a new output directory")
                    frame = read_csv(csv, symbol)
                    frame.attrs = saved["attrs"]
                else:
                    frame = fetch_pair_history(symbol, start, end, raw_path=directory / f"{symbol}.provider-raw.json") if profile.asset_class == "crypto" else fetch_equity_history(instrument, start, end, raw_path=directory / f"{symbol}.provider-raw.csv")
                result = coverage(frame, profile, instrument, start, end)
                boundary = BOUNDARIES.get(symbol)
                result.update(symbol=symbol, name=instrument.name, group=instrument.group, csv=str(csv),
                              request=request, attrs=frame.attrs,
                              security_boundary={"date": boundary[0], "note": boundary[1], "sources": boundary[2]} if boundary else CONTINUITY.get(symbol),
                              short_history_reason=(boundary[1] if boundary else "Exact Binance USDT spot pair began " + PAIR_BOUNDARIES[symbol][0]) if result["span_years"] < 10 and (boundary or symbol in PAIR_BOUNDARIES) else None)
                frame.to_csv(csv, index_label="Date")
                _write(sidecar, result)
                entries[symbol] = result
                print(f"{profile.id} {symbol}: {result['bars']} bars {result['first']}..{result['last']} missing={result['missing_count']}", flush=True)
            except Exception as exc:
                entries[symbol] = {"symbol": symbol, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                print(f"{profile.id} {symbol} FAILED: {entries[symbol]['error']}", flush=True)
            _write(output / "manifest.json", manifest)
            _write(directory / "metadata.json", entries)
    rows = ["# Long real-history coverage", "", f"Request {start} ≤ date < {end}", "", "| Market | Symbol | First | Last | Bars | Years | Missing | Status |", "|---|---|---|---|---:|---:|---:|---|"]
    for profile, entries in manifest["profiles"].items():
        for symbol, item in entries.items():
            rows.append(f"| {profile} | {symbol} | {item.get('first', '—')} | {item.get('last', '—')} | {item.get('bars', 0)} | {item.get('span_years', 0)} | {item.get('missing_count', '—')} | {item['status']} |")
    rows += ["", *[f"- {x}" for x in LIMITATIONS], "", "## Individual boundaries and missing sessions"]
    for entries in manifest["profiles"].values():
        for symbol, item in entries.items():
            if item.get("error"):
                rows += [f"- {symbol}: FAILED {item['error']}"]
            if item.get("short_history_reason"):
                rows += [f"- {symbol}: below 10 years — {item['short_history_reason']}"]
            boundary = item.get("security_boundary") or {}
            if boundary:
                rows += [f"- {symbol}: {boundary.get('note')} " + " ".join(boundary.get("sources", []))]
            for repair in item.get("attrs", {}).get("official_archive_repairs", []):
                rows += [f"- {symbol} {repair['date']}: REST shortened close-time interval; exact-day official monthly archive has valid UTC bounds; OHLCV unchanged={repair['ohlcv_equal']}. Exchange outage means limited actual trading within this day; do not infer 24h uninterrupted trading. {repair.get('maintenance_context_source') or ''}"]
            fixes = item.get("attrs", {}).get("adjustment_float_roundoff", [])
            if fixes:
                rows += [f"- {symbol}: {len(fixes)} documented Yahoo adjustment arithmetic corrections <=8 ULP; original unadjusted endpoint equality confirmed; raw and before/after in metadata."]
            if item.get("missing_dates"):
                rows += [f"- {symbol}: unresolved source absence / possible halt (not established): " + ", ".join(item["missing_dates"])]
    (output / "coverage.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def verify_existing_roundoff(output):
    """Attach original unadjusted OHLC equality evidence to prior checkpoints."""
    import yfinance as yf
    directory = Path(output) / "us_equities" / "data"
    for symbol in ("MSFT", "TSM", "ASML", "ORCL", "SPY"):
        sidecar = directory / f"{symbol}.json"
        item = json.loads(sidecar.read_text(encoding="utf-8"))
        fixes = item["attrs"].get("adjustment_float_roundoff", [])
        if not fixes:
            continue
        original = yf.download(symbol, start=item["request"]["start"], end=item["request"]["end"],
                               auto_adjust=False, progress=False, threads=False, timeout=30)
        original.to_csv(directory / f"{symbol}.unadjusted-raw.csv", index_label="Date")
        f = original.xs(symbol, axis=1, level=1)
        for fix in fixes:
            date, side = pd.Timestamp(fix["date"]), fix["field"]
            endpoint = next((c for c in ("Open", "Close") if f.loc[date, c] == f.loc[date, side]), None)
            if endpoint is None:
                raise ValueError(f"{symbol} {date}: no raw equality proof for roundoff")
            fix.update(original_equal_endpoint=endpoint, original_price=float(f.loc[date, side]))
        _write(sidecar, item)
        print(f"{symbol}: raw equality confirmed for {len(fixes)} float-roundoff corrections", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/long-history/20260930"))
    parser.add_argument("--start", default="2006-09-30")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--verify-roundoff", action="store_true")
    args = parser.parse_args()
    if args.verify_roundoff:
        verify_existing_roundoff(args.output)
    delattr(args, "verify_roundoff")
    manifest = prepare_history(**vars(args))
    return int(any(x["status"] == "failed" for p in manifest["profiles"].values() for x in p.values()))


if __name__ == "__main__":
    raise SystemExit(main())
