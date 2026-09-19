import knowledge as rk

cb = rk.CachedBucket()
names = cb.list_blob_names("raw/tracinginsights/2018/")
race_folders = sorted(set(n.split("/")[3] for n in names if len(n.split("/")) > 3))
for r in race_folders:
    print(r)