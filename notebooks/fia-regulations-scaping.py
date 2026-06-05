import pdfplumber
import re
import json

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
REGULATION_FILES = {
    2018: r"data\external\fia-pdfs\fia-sporting-regulations-2018.pdf",
    2019: r"data\external\fia-pdfs\fia-sporting-regulations-2019.pdf",
    2020: r"data\external\fia-pdfs\fia-sporting-regulations-2020.pdf",
    2021: r"data\external\fia-pdfs\fia-sporting-regulations-2021.pdf",
    2022: r"data\external\fia-pdfs\fia-sporting-regulations-2022.pdf",
    2023: r"data\external\fia-pdfs\fia-sporting-regulations-2023.pdf",
    2024: r"data\external\fia-pdfs\fia-sporting-regulations-2024.pdf",
    2025: r"data\external\fia-pdfs\fia-sporting-regulations-2025.pdf",
    2026: r"data\external\fia-pdfs\fia-sporting-regulations-2026.pdf",
}

OUTPUT_FILE = "f1_tyre_constraints.json"

# ─────────────────────────────────────────────
#  KNOWN GROUND TRUTH
#  Verified against FIA PDFs + Autosport/Pirelli sources.
#
#  Key change points:
#  - Sprint introduced: 2021 (12 dry vs 13 std)
#  - Q2 rule dropped: 2022
#  - Inter 4→5, Wet 3→2: 2024 (confirmed Autosport Apr 2024)
#  - 2021–2023 sprint: same inter/wet as standard GP (4/3)
#  - 2024–2025 sprint: same inter/wet as standard GP (5/2)
#  - 2026: back to 4 inter / 3 wet (new reg cycle, per Mercedes notes)
#  - Monaco exception (3 wet always): not encoded here as it's
#    circuit-specific, handled in race-level logic not season constraints
# ─────────────────────────────────────────────
KNOWN_VALUES = {
    #        std_dry  std_inter  std_wet  q2     wet_exc  sprint  sp_dry  sp_inter  sp_wet
    2018: dict(total_sets_allocated=13, intermediate_sets_allocated=4,  wet_sets_allocated=3, q2_start_tyre_rule=True,  wet_race_exception=True,  sprint_weekend=False, sprint_dry_sets=None, sprint_intermediate_sets=None, sprint_wet_sets=None),
    2019: dict(total_sets_allocated=13, intermediate_sets_allocated=4,  wet_sets_allocated=3, q2_start_tyre_rule=True,  wet_race_exception=True,  sprint_weekend=False, sprint_dry_sets=None, sprint_intermediate_sets=None, sprint_wet_sets=None),
    2020: dict(total_sets_allocated=13, intermediate_sets_allocated=4,  wet_sets_allocated=3, q2_start_tyre_rule=True,  wet_race_exception=True,  sprint_weekend=False, sprint_dry_sets=None, sprint_intermediate_sets=None, sprint_wet_sets=None),
    2021: dict(total_sets_allocated=13, intermediate_sets_allocated=4,  wet_sets_allocated=3, q2_start_tyre_rule=True,  wet_race_exception=True,  sprint_weekend=True,  sprint_dry_sets=12,   sprint_intermediate_sets=4,   sprint_wet_sets=3),
    2022: dict(total_sets_allocated=13, intermediate_sets_allocated=4,  wet_sets_allocated=3, q2_start_tyre_rule=False, wet_race_exception=True,  sprint_weekend=True,  sprint_dry_sets=12,   sprint_intermediate_sets=4,   sprint_wet_sets=3),
    2023: dict(total_sets_allocated=13, intermediate_sets_allocated=4,  wet_sets_allocated=3, q2_start_tyre_rule=False, wet_race_exception=True,  sprint_weekend=True,  sprint_dry_sets=12,   sprint_intermediate_sets=4,   sprint_wet_sets=3),
    2024: dict(total_sets_allocated=13, intermediate_sets_allocated=5,  wet_sets_allocated=2, q2_start_tyre_rule=False, wet_race_exception=True,  sprint_weekend=True,  sprint_dry_sets=12,   sprint_intermediate_sets=5,   sprint_wet_sets=2),
    2025: dict(total_sets_allocated=13, intermediate_sets_allocated=5,  wet_sets_allocated=2, q2_start_tyre_rule=False, wet_race_exception=True,  sprint_weekend=True,  sprint_dry_sets=12,   sprint_intermediate_sets=5,   sprint_wet_sets=2),
    2026: dict(total_sets_allocated=13, intermediate_sets_allocated=4,  wet_sets_allocated=3, q2_start_tyre_rule=False, wet_race_exception=True,  sprint_weekend=True,  sprint_dry_sets=12,   sprint_intermediate_sets=4,   sprint_wet_sets=3),
}


# ─────────────────────────────────────────────
#  WORD → INT
# ─────────────────────────────────────────────
def text_to_int(text: str) -> int:
    word_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16,
    }
    cleaned = re.sub(r"[()]", "", text).strip().lower()
    if cleaned in word_map:
        return word_map[cleaned]
    try:
        return int(cleaned)
    except ValueError:
        return 0


# ─────────────────────────────────────────────
#  PARSE PDF INTO ARTICLES
# ─────────────────────────────────────────────
def parse_regulations(year: int, file_path: str) -> dict:
    print(f"\n📚 Parsing Year: {year}...")

    if year == 2026:
        article_pattern = re.compile(r"^\s*(B\d+)(?:\.\d+)?", re.IGNORECASE)
    else:
        article_pattern = re.compile(
            r"^\s*(?:ARTICLE\s+(\d+)|(\d+)[\.\)]\s+[A-Z])", re.IGNORECASE
        )

    articles = {}
    current_article = None

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                for line in text.split("\n"):
                    match = article_pattern.match(line)
                    if match:
                        new_key = match.group(1).upper() if year == 2026 else (match.group(1) or match.group(2))
                        if new_key:
                            current_article = new_key
                            if current_article not in articles:
                                articles[current_article] = ""
                            articles[current_article] += line + "\n"
                    elif current_article:
                        articles[current_article] += line + "\n"

    except FileNotFoundError:
        print(f"   ❌ File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return {}

    return articles


# ─────────────────────────────────────────────
#  FIND TYRE ARTICLE
# ─────────────────────────────────────────────
def find_tyre_article_id(articles_dict: dict) -> str | None:
    header_keywords = [
        "supply of tyres", "quantity of tyres", "use of tyres",
        "tyre allocation", "tyres allocated", "tyre limitation",
    ]
    body_signals = [
        r"sets of dry.weather tyre", r"no driver may use more than",
        r"sets of intermediate", r"sets of wet.weather",
        r"dry.weather tyre", r"intermediate tyre",
        r"tyre manufacturer", r"pirelli",
    ]

    best_id, best_score = None, 0
    for art_id, content in articles_dict.items():
        header = content[:3000].lower()
        body = content.lower()
        if not any(k in header for k in header_keywords):
            continue
        score = sum(1 for sig in body_signals if re.search(sig, body, re.IGNORECASE))
        if score > best_score:
            best_score = score
            best_id = art_id

    return best_id


# ─────────────────────────────────────────────
#  EXTRACT FROM ARTICLE TEXT
#  Only mandatory_dry_compounds is extracted via
#  regex — everything else is set by KNOWN_VALUES
#  since PDF phrasing varies too much year-to-year
#  and the values are well-documented externally.
# ─────────────────────────────────────────────
def extract_tyre_data(article_text: str) -> dict:
    data = {
        "mandatory_dry_compounds": 0,
        "total_sets_allocated": 0,
        "intermediate_sets_allocated": 0,
        "wet_sets_allocated": 0,
        "sprint_weekend": False,
        "sprint_dry_sets": None,
        "sprint_intermediate_sets": None,
        "sprint_wet_sets": None,
        "wet_race_exception": False,
        "q2_start_tyre_rule": False,
    }

    # ── 1. MANDATORY COMPOUNDS ────────────────────────────────────────────
    m = re.search(
        r"must\s+use\s+at\s+least\s+(\w+|\d+)(?:\s*\(\d+\))?\s+different",
        article_text, re.IGNORECASE | re.DOTALL
    )
    if m:
        data["mandatory_dry_compounds"] = text_to_int(m.group(1))

    # ── 2. DRY SET ALLOCATION (regex, validated later) ────────────────────
    # Must score >= 10 to be plausible as a dry count.
    # Anchored to "dry" where possible to avoid matching inter/wet counts.
    dry_patterns = [
        re.compile(r"no\s+driver\s+may\s+use\s+more\s+than\s+(\w+|\d+)(?:\s*\(\d+\))?\s+sets\s+of\s+dry", re.IGNORECASE | re.DOTALL),
        re.compile(r"allocated\s+(\w+|\d+)(?:\s*\(\d+\))?\s+sets\s+of\s+dry", re.IGNORECASE | re.DOTALL),
        re.compile(r"no\s+more\s+than\s+(\w+|\d+)(?:\s*\(\d+\))?\s+sets\s+of\s+dry", re.IGNORECASE | re.DOTALL),
        re.compile(r"no\s+driver\s+may\s+use\s+more\s+than\s+(\w+|\d+)(?:\s*\(\d+\))?\s+sets", re.IGNORECASE | re.DOTALL),
        re.compile(r"(\w+|\d+)\s*\(\d+\)\s+sets\s+of\s+(?:dry|slick)", re.IGNORECASE | re.DOTALL),
    ]
    for p in dry_patterns:
        m = p.search(article_text)
        if m:
            val = text_to_int(m.group(1))
            if val >= 10:
                data["total_sets_allocated"] = val
                break

    # ── 3. WET RACE EXCEPTION ─────────────────────────────────────────────
    wet_exc_patterns = [
        re.compile(r"unless.*?(?:intermediate|wet.weather)\s+tyre", re.IGNORECASE | re.DOTALL),
        re.compile(r"wet.weather\s+tyre.*?exempt", re.IGNORECASE | re.DOTALL),
        re.compile(r"if\s+(?:intermediate|wet)\s+tyre.*?not\s+required", re.IGNORECASE | re.DOTALL),
    ]
    for p in wet_exc_patterns:
        if p.search(article_text):
            data["wet_race_exception"] = True
            break

    # ── 4. Q2 TYRE START RULE ─────────────────────────────────────────────
    q2_checks = [
        re.compile(r"start\s+the\s+race\s+on\s+the\s+tyre", re.IGNORECASE),
        re.compile(r"fastest\s+time\s+in\s+(?:Q2|the\s+second)", re.IGNORECASE),
        re.compile(r"fitted\s+with\s+the\s+tyre.*?Q2", re.IGNORECASE | re.DOTALL),
        re.compile(r"tyre.*?set.*?fastest.*?Q2", re.IGNORECASE | re.DOTALL),
        re.compile(r"qualified\s+for\s+Q3\s+must\s+be\s+fitted", re.IGNORECASE),
    ]
    if sum(1 for p in q2_checks if p.search(article_text)) >= 2:
        data["q2_start_tyre_rule"] = True

    return data


# ─────────────────────────────────────────────
#  APPLY & VALIDATE KNOWN VALUES
#  Regex is used as a cross-check; KNOWN_VALUES
#  are authoritative. Logs any mismatch so you
#  can spot if a PDF changes phrasing.
# ─────────────────────────────────────────────
def apply_known_values(year: int, extracted: dict) -> dict:
    known = KNOWN_VALUES[year]
    result = extracted.copy()

    # Fields always set from known values (too variable to regex reliably)
    always_override = [
        "intermediate_sets_allocated", "wet_sets_allocated",
        "sprint_weekend", "sprint_dry_sets",
        "sprint_intermediate_sets", "sprint_wet_sets",
    ]
    for field in always_override:
        result[field] = known[field]

    # Dry sets: override only if regex got something implausible
    if result["total_sets_allocated"] < 10:
        print(f"      ⚠️  Dry sets extracted as {result['total_sets_allocated']} "
              f"— overriding with known value: {known['total_sets_allocated']}")
        result["total_sets_allocated"] = known["total_sets_allocated"]
    elif result["total_sets_allocated"] != known["total_sets_allocated"]:
        print(f"      ⚠️  Dry sets mismatch: regex={result['total_sets_allocated']}, "
              f"known={known['total_sets_allocated']} — using known")
        result["total_sets_allocated"] = known["total_sets_allocated"]

    # wet_race_exception: override if regex missed it
    if not result["wet_race_exception"] and known["wet_race_exception"]:
        print(f"      ⚠️  wet_race_exception missed by regex — overriding to True")
        result["wet_race_exception"] = True

    # q2 rule: log mismatch, trust known
    if result["q2_start_tyre_rule"] != known["q2_start_tyre_rule"]:
        print(f"      ⚠️  q2_start_tyre_rule mismatch "
              f"(regex={result['q2_start_tyre_rule']}, known={known['q2_start_tyre_rule']}) "
              f"— using known")
        result["q2_start_tyre_rule"] = known["q2_start_tyre_rule"]

    return result


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    full_database = {}
    print("\n🚀 Starting FIA Tyre Regulation Extractor...")

    for year, file_path in REGULATION_FILES.items():
        regs = parse_regulations(year, file_path)
        if not regs:
            print(f"   ⚠️  {year}: No articles parsed — check file path.")
            continue

        tyre_id = find_tyre_article_id(regs)
        if not tyre_id:
            print(f"   ❌ {year}: Could not find tyre article")
            print(f"      Available articles: {list(regs.keys())[:20]}")
            continue

        print(f"   🎯 {year}: Tyre rules found in Article {tyre_id}")
        extracted  = extract_tyre_data(regs[tyre_id])
        validated  = apply_known_values(year, extracted)
        validated["source_article"] = tyre_id
        full_database[str(year)] = validated

        if validated["mandatory_dry_compounds"] == 0:
            print(f"      ⚠️  mandatory_dry_compounds = 0 — check article content")
            print(f"      📋 First 800 chars:\n{regs[tyre_id][:800]}\n")
        else:
            print(f"      ✅ {json.dumps(validated)}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(full_database, f, indent=4)

    print(f"\n✅ Done. Saved to {OUTPUT_FILE}")
    print(f"\n{'Year':<6} {'Dry':>4} {'Inter':>6} {'Wet':>4} {'Sp.Dry':>7} {'Sp.Int':>7} {'Sp.Wet':>7} {'Q2':>5} {'WetExc':>7}")
    print("-" * 60)
    for year, d in full_database.items():
        print(
            f"{year:<6} {d['total_sets_allocated']:>4} {d['intermediate_sets_allocated']:>6} "
            f"{d['wet_sets_allocated']:>4} {str(d['sprint_dry_sets']):>7} "
            f"{str(d['sprint_intermediate_sets']):>7} {str(d['sprint_wet_sets']):>7} "
            f"{str(d['q2_start_tyre_rule']):>5} {str(d['wet_race_exception']):>7}"
        )


if __name__ == "__main__":
    main()