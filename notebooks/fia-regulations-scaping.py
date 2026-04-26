import pdfplumber
import re
import json

REGULATION_FILES = {
    2018: r"fia-pdfs\fia-sporting-regulations-2018.pdf",
    2019: r"fia-pdfs\fia-sporting-regulations-2019.pdf",
    2020: r"fia-pdfs\fia-sporting-regulations-2020.pdf",
    2021: r"fia-pdfs\fia-sporting-regulations-2021.pdf",
    2022: r"fia-pdfs\fia-sporting-regulations-2022.pdf",
    2023: r"fia-pdfs\fia-sporting-regulations-2023.pdf",
    2024: r"fia-pdfs\fia-sporting-regulations-2024.pdf",
    2025: r"fia-pdfs\fia-sporting-regulations-2025.pdf",
    2026: r"fia-pdfs\fia-sporting-regulations-2026.pdf"

}
def text_to_int(text):
    """Converts 'thirteen' or '13' to integer."""
    mapping = {
        "one": 1, "two": 2, "three": 3, "four": 4, 
        "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        "thirteen": 13, "fourteen": 14,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
        "8": 8, "12": 12, "13": 13
    }
    clean_text = re.sub(r"[()]", "", text).lower()
    return mapping.get(clean_text, 0)

# ---------------------------------------------------------
# 3. PARSER
# ---------------------------------------------------------
def parse_regulations(year, file_path):
    print(f"\n📚 Parsing Year: {year}...")
    
    # 2026 uses "Section B" logic
    if year == 2026:
        # Regex to catch "B6.1" or "ARTICLE B6"
        article_pattern = r"^\s*(?:ARTICLE\s+)?(B\d+(?:\.\d+)?)"
    else:
        # Regex to catch "ARTICLE 24" or "24. SUPPLY"
        article_pattern = r"^\s*(?:ARTICLE\s+(\d+)|(\d+)[\.\)]\s+[A-Z])"
    
    articles = {}
    current_article = None
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text: continue
                
                for line in text.split("\n"):
                    match = re.match(article_pattern, line, re.IGNORECASE)
                    
                    if match:
                        new_key = match.group(1) if match.group(1) else match.group(2)
                        if new_key:
                            current_article = new_key
                            if current_article not in articles:
                                articles[current_article] = ""
                            articles[current_article] += line + "\n"
                    
                    elif current_article:
                        articles[current_article] += line + "\n"
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return {}
    
    return articles

# ---------------------------------------------------------
# 4. SEEKER
# ---------------------------------------------------------
def find_tyre_article_id(articles_dict):
    keywords = ["supply of tyres", "quantity of tyres", "use of tyres"]
    for art_id, content in articles_dict.items():
        header = content[:1000].lower()
        if any(k in header for k in keywords):
            return art_id
    return None

# ---------------------------------------------------------
# 5. EXTRACTOR (MULTI-STRATEGY)
# ---------------------------------------------------------
def extract_tyre_data(article_text):
    data = {
        "mandatory_dry_compounds": 0,
        "total_sets_allocated": 0,
        "wet_race_exception": False,
        "q2_start_tyre_rule": False
    }
    
    # --- 1. MANDATORY USAGE ---
    usage_pattern = r"must\s+use\s+at\s+least\s+(\w+|\d+)(?:\s*\(\d+\))?\s+different"
    match_usage = re.search(usage_pattern, article_text, re.IGNORECASE | re.DOTALL)
    if match_usage:
        data["mandatory_dry_compounds"] = text_to_int(match_usage.group(1))
    
    # --- 2. ALLOCATION (The Hard Part) ---
    # Strategy A: "allocated 13 sets" (Common 2021-2024)
    # Strategy B: "use no more than 13 sets" (Common 2018-2020)
    # Strategy C: "Eight (8) sets" (2026 specific structure)
    
    alloc_patterns = [
        r"allocated\s+(\w+|\d+)(?:\s*\(\d+\))?\s+sets",        # Strategy A
        r"use\s+no\s+more\s+than\s+(\w+|\d+)\s+sets",           # Strategy B
        r"allocated\s+.*?(\w+|\d+)(?:\s*\(\d+\))?\s+sets"       # Strategy C (Broad search for 2026)
    ]
    
    for p in alloc_patterns:
        match = re.search(p, article_text, re.IGNORECASE | re.DOTALL)
        if match:
            val = text_to_int(match.group(1))
            if val > 0:
                data["total_sets_allocated"] = val
                break # Stop if we found a valid number

    # --- 3. WET RACE EXCEPTION ---
    wet_pattern = r"Unless\s+he\s+has\s+used\s+intermediate\s+or\s+wet-weather\s+tyres"
    if re.search(wet_pattern, article_text, re.IGNORECASE):
        data["wet_race_exception"] = True

    # --- 4. Q2 RULE ---
    q2_pattern = r"start\s+the\s+race.*?tyres.*?fastest\s+time.*?during\s+(?:Q2|the\s+second\s+period)"
    if re.search(q2_pattern, article_text, re.IGNORECASE | re.DOTALL):
        data["q2_start_tyre_rule"] = True
        
    return data

# ---------------------------------------------------------
# 6. EXECUTION
# ---------------------------------------------------------
full_database = {}
print("\n🚀 Starting Universal F1 Scraper V3.0...")

for year, file_path in REGULATION_FILES.items():
    regs = parse_regulations(year, file_path)
    tyre_id = find_tyre_article_id(regs)
    
    if tyre_id:
        print(f"   🎯 {year}: Found Tyre Rules in Article {tyre_id}")
        constraints = extract_tyre_data(regs[tyre_id])
        constraints["source_article"] = tyre_id
        
        full_database[str(year)] = constraints
        
        # Validation Check
        if constraints["total_sets_allocated"] == 0:
            print(f"      ⚠️ WARNING: Allocation count is still 0. Check Regex!")
        else:
            print(f"      ✅ Extracted: {json.dumps(constraints)}")
    else:
        print(f"   ❌ {year}: Could not identify Tyre section.")

# Save
with open("f1_tyre_constraints.json", "w") as f:
    json.dump(full_database, f, indent=4)