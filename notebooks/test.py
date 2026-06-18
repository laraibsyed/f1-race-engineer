import fastf1

CACHE_DIR = "data\\raw"
fastf1.Cache.enable_cache(CACHE_DIR)

schedule = fastf1.get_event_schedule(2026)
schedule = schedule[schedule['EventFormat'] != 'testing']

for _, event in schedule.iterrows():
    print(f"Round {event['RoundNumber']:2} | {event['EventName']:<35} | {event['EventFormat']}")

# import fastf1

# CACHE_DIR = "data\\raw"
# fastf1.Cache.enable_cache(CACHE_DIR)

# event = fastf1.get_event(2023, 4)
# print(event)