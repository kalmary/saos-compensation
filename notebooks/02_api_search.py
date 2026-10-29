"""Run with: uv run notebooks/02_api_search.py (or execute the # %% cells)."""

# %% Search one page
from pprint import pprint

import requests

params = {
    "all": "zadośćuczynienie",
    "courtType": "COMMON",
    "judgmentTypes": "SENTENCE",
    "pageSize": 10,
    "pageNumber": 0,
    "sortingField": "DATABASE_ID",
    "sortingDirection": "ASC",
}
response = requests.get(
    "https://www.saos.org.pl/api/search/judgments", params=params, timeout=30
)
response.raise_for_status()
results = response.json()
assert isinstance(results["items"], list)
assert type(results["info"]["totalResults"]) is int
print(response.url)
print("Search matches:", results["info"]["totalResults"])
for item in results["items"]:
    print(item["id"], item["judgmentDate"], item["courtCases"])

# %% Inspect the response structure
pprint(results["queryTemplate"])
if not results["items"]:
    raise SystemExit("No matches. Change params and run the search again.")
pprint(results["items"][0])

# %% Fetch one full judgment; change the index to inspect another result
judgment_id = results["items"][0]["id"]
response = requests.get(
    f"https://www.saos.org.pl/api/judgments/{judgment_id}", timeout=30
)
response.raise_for_status()
judgment = response.json()["data"]
assert judgment["id"] == judgment_id
assert isinstance(judgment["textContent"], str) and judgment["textContent"].strip()
text = judgment["textContent"]
print(f"https://www.saos.org.pl/judgments/{judgment_id}")
print(judgment.keys())
print(text[:3000])
