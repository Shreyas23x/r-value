"""R-Value configuration: seasons, CBA constants, model weights, ground truth."""
import json
from pathlib import Path

SEASON = "2025-26"               # current (most recently completed) season
# 2019-20 (bubble; uneven GP) and 2020-21 (72 games; tax pro-rated at
# settlement) are deliberately excluded: they distort both availability
# and the cost model.
HISTORY_SEASONS = ["2018-19", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
PREDICT_SEASON = "2026-27"

def hh_season(season: str) -> int:
    """hoopshype labels a season by its starting year: 2025-26 -> 2025."""
    return int(season[:4])

# ---- CBA constants per league year (USD) ----
# rates: luxury-tax multipliers for the first four brackets, then +step each.
# Pre-2025-26 uses the 2017 CBA schedule (repeater = +1.00 flat on each rate);
# 2025-26 onward uses the 2023 CBA schedule with bracket width indexed to cap
# growth. 2026-27 figures are league projections (~7% growth), flagged estimated.
CBA = {
    # pre-2023-CBA years: apron2 mirrors apron1 (no second apron existed)
    "2018-19": dict(cap=101_869_000, tax=123_733_000, apron1=129_817_000,
                    apron2=129_817_000, rates_std=[1.50, 1.75, 2.50, 3.25],
                    rates_rep=[2.50, 2.75, 3.50, 4.25], step=0.50,
                    width=5_000_000, repeaters={"CLE"}, estimated=False),
    "2021-22": dict(cap=112_414_000, tax=136_606_000, apron1=143_002_000,
                    apron2=143_002_000, rates_std=[1.50, 1.75, 2.50, 3.25],
                    rates_rep=[2.50, 2.75, 3.50, 4.25], step=0.50,
                    width=5_000_000, repeaters={"GSW"}, estimated=False),
    "2022-23": dict(cap=123_655_000, tax=150_267_000, apron1=156_983_000,
                    apron2=156_983_000, rates_std=[1.50, 1.75, 2.50, 3.25],
                    rates_rep=[2.50, 2.75, 3.50, 4.25], step=0.50,
                    width=5_000_000, repeaters={"GSW"}, estimated=False),
    "2023-24": dict(cap=136_021_000, tax=165_294_000, apron1=172_346_000,
                    apron2=182_794_000, rates_std=[1.50, 1.75, 2.50, 3.25],
                    rates_rep=[2.50, 2.75, 3.50, 4.25], step=0.50,
                    width=5_000_000, repeaters={"GSW", "LAC"}, estimated=False),
    "2024-25": dict(cap=140_588_000, tax=170_814_000, apron1=178_132_000,
                    apron2=188_931_000, rates_std=[1.50, 1.75, 2.50, 3.25],
                    rates_rep=[2.50, 2.75, 3.50, 4.25], step=0.50,
                    width=5_000_000, repeaters={"GSW", "LAC", "PHX", "MIL"},
                    estimated=False),
    "2025-26": dict(cap=154_647_000, tax=187_895_000, apron1=195_946_000,
                    apron2=207_824_000, rates_std=[1.00, 1.25, 3.50, 4.75],
                    rates_rep=[3.00, 3.25, 5.50, 6.75], step=0.50,
                    width=5_685_000,
                    repeaters={"GSW", "LAC", "MIL", "BOS", "PHX", "LAL", "DEN"},
                    estimated=False),
    "2026-27": dict(cap=165_472_000, tax=201_048_000, apron1=209_662_000,
                    apron2=222_372_000, rates_std=[1.00, 1.25, 3.50, 4.75],
                    rates_rep=[3.00, 3.25, 5.50, 6.75], step=0.50,
                    width=6_083_000,
                    repeaters={"GSW", "MIL", "BOS", "PHX", "LAL", "DEN", "MIN", "NYK", "CLE"},
                    estimated=True),
}

MIN_SALARY = 1_272_870           # 0-YOS veteran minimum (2025-26)
# minimum-salary scale factors by years of service (0..10+), 2025-26 proportions
MIN_SALARY_FACTORS = [1.00, 1.61, 1.80, 1.87, 1.93, 2.10, 2.24, 2.38, 2.52, 2.53, 2.85]

# ---- Model weights (defaults; calibrated values override via weights file) ----
DEFAULT_WEIGHTS = dict(box=0.30, eff=0.20, off=0.15, dfn=0.15, adv=0.20)
WEIGHTS_FILE = Path(__file__).resolve().parent.parent / "output" / "calibrated_weights.json"

def load_weights() -> dict:
    if WEIGHTS_FILE.exists():
        return json.loads(WEIGHTS_FILE.read_text())["weights"]
    return DEFAULT_WEIGHTS

def load_star_lambda() -> float:
    if WEIGHTS_FILE.exists():
        return json.loads(WEIGHTS_FILE.read_text()).get("star_lambda", 0.0)
    return 0.0

def load_rot_lambda() -> float:
    if WEIGHTS_FILE.exists():
        return json.loads(WEIGHTS_FILE.read_text()).get("rot_lambda", 0.0)
    return 0.0

QUALIFIED_MIN = 500          # minutes threshold for z-score calibration pool
SHRINK_MIN = 500             # reliability shrinkage constant for impact ratings
FULL_CREDIT_MIN = 2200       # minutes for full availability credit
AVAIL_FLOOR = 0.55           # WPS = PS * (floor + (1-floor)*A)
COST_EXPONENT = 0.35         # dampening of cost leverage in player R-Value
COST_SHARE_CLIP = (0.002, 0.40)   # CostIndex clip range
GP_ELIGIBLE = 65             # award-eligibility games threshold (2023 CBA)

def gp_threshold(season: str | None) -> int:
    """65-game award rule exists only from 2023-24; earlier seasons use a
    soft 40-game floor (voters de facto required substantial participation)."""
    if season is None or season >= "2023-24":
        return GP_ELIGIBLE
    return 40

# Box-production per-36 coefficients
BOX_WEIGHTS = {"PTS": 1.00, "REB": 0.70, "AST": 1.40, "STL": 2.20, "BLK": 2.00, "TOV": -1.40}

# ---- Ground truth: All-NBA selections (nba.com official announcements) ----
ALL_NBA = {
    "2018-19": ["Giannis Antetokounmpo", "James Harden", "Stephen Curry",
                "Paul George", "Nikola Jokic", "Joel Embiid", "Kevin Durant",
                "Damian Lillard", "Kawhi Leonard", "Kyrie Irving",
                "Russell Westbrook", "Blake Griffin", "LeBron James",
                "Rudy Gobert", "Kemba Walker"],
    "2021-22": ["Giannis Antetokounmpo", "Devin Booker", "Luka Doncic",
                "Nikola Jokic", "Jayson Tatum", "Ja Morant", "Kevin Durant",
                "Joel Embiid", "DeMar DeRozan", "Stephen Curry",
                "Karl-Anthony Towns", "LeBron James", "Chris Paul",
                "Trae Young", "Pascal Siakam"],
    "2022-23": ["Shai Gilgeous-Alexander", "Luka Doncic", "Joel Embiid",
                "Jayson Tatum", "Giannis Antetokounmpo", "Stephen Curry",
                "Donovan Mitchell", "Jimmy Butler", "Nikola Jokic",
                "Jaylen Brown", "Damian Lillard", "De'Aaron Fox",
                "LeBron James", "Julius Randle", "Domantas Sabonis"],
    "2023-24": ["Giannis Antetokounmpo", "Shai Gilgeous-Alexander", "Luka Doncic",
                "Nikola Jokic", "Jayson Tatum", "Jalen Brunson", "Anthony Davis",
                "Kevin Durant", "Anthony Edwards", "Kawhi Leonard", "Devin Booker",
                "Stephen Curry", "Tyrese Haliburton", "LeBron James", "Domantas Sabonis"],
    "2024-25": ["Shai Gilgeous-Alexander", "Nikola Jokic", "Giannis Antetokounmpo",
                "Donovan Mitchell", "Jayson Tatum", "LeBron James", "Anthony Edwards",
                "Evan Mobley", "Stephen Curry", "Jalen Brunson", "Cade Cunningham",
                "Tyrese Haliburton", "James Harden", "Karl-Anthony Towns", "Jalen Williams"],
    "2025-26": ["Shai Gilgeous-Alexander", "Victor Wembanyama", "Cade Cunningham",
                "Luka Doncic", "Nikola Jokic", "Jaylen Brown", "Jalen Brunson",
                "Kevin Durant", "Kawhi Leonard", "Donovan Mitchell", "Tyrese Maxey",
                "Jamal Murray", "Jalen Johnson", "Chet Holmgren", "Jalen Duren"],
}

CONFERENCE = {
    "ATL": "East", "BOS": "East", "BKN": "East", "CHA": "East", "CHI": "East",
    "CLE": "East", "DET": "East", "IND": "East", "MIA": "East", "MIL": "East",
    "NYK": "East", "ORL": "East", "PHI": "East", "TOR": "East", "WAS": "East",
    "DAL": "West", "DEN": "West", "GSW": "West", "HOU": "West", "LAC": "West",
    "LAL": "West", "MEM": "West", "MIN": "West", "NOP": "West", "OKC": "West",
    "PHX": "West", "POR": "West", "SAC": "West", "SAS": "West", "UTA": "West",
}

# hoopshype teamID -> NBA abbreviation
HH_TEAM_ABBR = {
    "1": "ATL", "2": "BOS", "17": "BKN", "5312": "CHA", "4": "CHI", "5": "CLE",
    "6": "DAL", "7": "DEN", "8": "DET", "9": "GSW", "10": "HOU", "11": "IND",
    "12": "LAC", "13": "LAL", "29": "MEM", "14": "MIA", "15": "MIL", "16": "MIN",
    "3": "NOP", "18": "NYK", "25": "OKC", "19": "ORL", "20": "PHI", "21": "PHX",
    "22": "POR", "23": "SAC", "24": "SAS", "28": "TOR", "26": "UTA", "27": "WAS",
}

HH_TEAM_SLUGS = {
    "atlanta-hawks": "1", "boston-celtics": "2", "brooklyn-nets": "17",
    "charlotte-hornets": "5312", "chicago-bulls": "4", "cleveland-cavaliers": "5",
    "dallas-mavericks": "6", "denver-nuggets": "7", "detroit-pistons": "8",
    "golden-state-warriors": "9", "houston-rockets": "10", "indiana-pacers": "11",
    "los-angeles-clippers": "12", "los-angeles-lakers": "13", "memphis-grizzlies": "29",
    "miami-heat": "14", "milwaukee-bucks": "15", "minnesota-timberwolves": "16",
    "new-orleans-pelicans": "3", "new-york-knicks": "18", "oklahoma-city-thunder": "25",
    "orlando-magic": "19", "philadelphia-76ers": "20", "phoenix-suns": "21",
    "portland-trail-blazers": "22", "sacramento-kings": "23", "san-antonio-spurs": "24",
    "toronto-raptors": "28", "utah-jazz": "26", "washington-wizards": "27",
}

# manual salary patches for players absent from all salary sources
# (unsigned 2026 FAs whose old records were purged); from public reporting
KNOWN_SALARIES = {
    ("lebron james", 2018): 35_654_150,
    ("lebron james", 2021): 41_180_544,
    ("lebron james", 2022): 44_474_988,
    ("james harden", 2018): 30_431_854,
    ("james harden", 2021): 44_310_840,
    ("james harden", 2022): 33_000_000,
    ("russell westbrook", 2018): 35_654_150,
    ("lebron james", 2023): 47_607_350,
    ("lebron james", 2024): 48_728_845,
    ("lebron james", 2025): 52_627_153,
    ("james harden", 2023): 35_640_000,
    ("james harden", 2024): 33_653_846,
}

# name-normalization aliases: hoopshype-normalized -> nba.com-normalized
NAME_ALIASES = {
    "nicolas claxton": "nic claxton",
    "cameron thomas": "cam thomas",
    "kenyon martin": "kj martin",
    "gregory jackson": "gg jackson",
    "herb jones": "herbert jones",
    "nah shon hyland": "bones hyland",
    "ron holland": "ronald holland",
    "carlton carrington": "bub carrington",
    "alexandre sarr": "alex sarr",
    "moe wagner": "moritz wagner",
    "dennis schroeder": "dennis schroder",
    "santiago aldama": "santi aldama",
    "sviatoslav mykhailiuk": "svi mykhailiuk",
}
