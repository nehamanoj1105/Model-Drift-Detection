"""Phase 3 -- verify every references.bib entry against Crossref by title.

Prints the best Crossref match (DOI, year, container) for each entry so the
bibliography can be checked without hand-asserting DOIs. Network required.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIB = os.path.join(ROOT, "paper", "references.bib")

entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", open(BIB).read(), re.S)
print(f"{'key':26s} {'Crossref title match':44s} {'DOI':30s} year")
for key, body in entries:
    t = re.search(r"title\s*=\s*[{\"](.+?)[}\"],\n", body, re.S)
    if not t:
        continue
    title = re.sub(r"\s+", " ", t.group(1)).replace("{", "").replace("}", "")
    title = title.replace("\\&", "&").replace("--", "-")
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 1})
    try:
        with urllib.request.urlopen(
                f"https://api.crossref.org/works?{q}", timeout=20) as r:
            items = json.load(r)["message"]["items"]
    except Exception as e:
        print(f"{key:26s} ERROR {e}")
        continue
    if not items:
        print(f"{key:26s} (no match) for {title[:40]}")
        continue
    it = items[0]
    mt = (it.get("title") or ["?"])[0]
    yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
    cont = (it.get("container-title") or ["?"])[0]
    print(f"{key:26s} {mt[:43]:44s} {it.get('DOI','?'):30s} {yr}  [{cont[:30]}]")
    time.sleep(0.3)
