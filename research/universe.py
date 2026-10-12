"""Fixed research cohorts, not a current market-cap ranking or trade list."""
from dataclasses import dataclass, field

from wyckoff_core.models import CoreConfig


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    group: str
    listed_since: str | None = None
    identity_tokens: tuple[str, ...] = ()
    identity_source: str | None = None


@dataclass(frozen=True)
class ResearchProfile:
    id: str
    label: str
    asset_class: str
    market_type: str
    price_currency: str
    instruments: tuple[Instrument, ...]
    benchmark: str
    timeframe: str = "1d"
    config: CoreConfig = field(default_factory=CoreConfig)


SPCX_SOURCE = "https://m.nasdaqtrader.com/TraderNews.aspx?id=DTN2026-8"
EQUITIES = ResearchProfile(
    "us_equities", "美股 · 科技七巨头与 AI 产业", "equity", "cash", "USD",
    (
        Instrument("AAPL", "Apple", "科技七巨头"),
        Instrument("MSFT", "Microsoft", "科技七巨头"),
        Instrument("NVDA", "NVIDIA", "科技七巨头"),
        Instrument("AMZN", "Amazon", "科技七巨头"),
        Instrument("GOOGL", "Alphabet Class A", "科技七巨头"),
        Instrument("META", "Meta Platforms", "科技七巨头"),
        Instrument("TSLA", "Tesla", "科技七巨头"),
        Instrument("MU", "Micron Technology", "存储与 HBM"),
        Instrument("SPCX", "Space Exploration Technologies Corp. Class A", "用户指定",
                   "2026-06-12", ("space exploration", "spacex"), SPCX_SOURCE),
        Instrument("AVGO", "Broadcom", "芯片与网络"),
        Instrument("AMD", "Advanced Micro Devices", "计算芯片"),
        Instrument("TSM", "Taiwan Semiconductor Manufacturing ADR", "芯片制造"),
        Instrument("ASML", "ASML Holding", "半导体设备"),
        Instrument("IBM", "International Business Machines", "企业 AI 与混合云"),
        Instrument("ORCL", "Oracle", "云与软件"),
        Instrument("SMCI", "Super Micro Computer", "服务器"),
        Instrument("INTC", "Intel", "AI 计算与芯片制造"),
        Instrument("ADBE", "Adobe", "生成式 AI 软件"),
        Instrument("ANET", "Arista Networks", "数据中心网络"),
    ), "SPY",
)
CRYPTO = ResearchProfile(
    "binance_spot", "数字货币 · Binance 现货", "crypto", "spot", "USDT",
    (Instrument("BTCUSDT", "Bitcoin / USDT", "现货"),
     Instrument("ETHUSDT", "Ether / USDT", "现货"),
     Instrument("DOGEUSDT", "Dogecoin / USDT", "现货")), "BTCUSDT",
)
PROFILES = {profile.id: profile for profile in (EQUITIES, CRYPTO)}


def verify_identity(instrument: Instrument, info: dict) -> dict:
    """Fail closed for reused symbols before accepting provider history."""
    if not instrument.identity_tokens:
        return {"status": "not_required"}
    names = " ".join(str(info.get(k, "")) for k in ("longName", "shortName")).lower()
    if (str(info.get("symbol", "")).upper() != instrument.symbol
            or info.get("quoteType") != "EQUITY"
            or any(token in names for token in ("tuttle", "spac and new issue", " etf"))
            or (info.get("cusip") and info["cusip"] != "84615Q103")
            or not any(token in names for token in instrument.identity_tokens)):
        raise ValueError(f"{instrument.symbol}: provider identity does not verify SpaceX equity; refusing reused ticker history")
    return {"status": "verified_provider_name_and_type", "symbol": info["symbol"],
            "name": info.get("longName") or info.get("shortName"), "quote_type": info["quoteType"],
            "listed_since": instrument.listed_since, "source": instrument.identity_source,
            "expected_cusip": "84615Q103", "provider_cusip": info.get("cusip")}
