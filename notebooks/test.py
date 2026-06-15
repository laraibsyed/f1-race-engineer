import fastf1

CACHE_DIR = "data\\raw"

fastf1.Cache.enable_cache(CACHE_DIR)

event = fastf1.get_event(2022, 4)
print(event['EventFormat'])