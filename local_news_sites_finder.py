import csv
import os
import requests
from dotenv import load_dotenv

load_dotenv()

DISCOVERED_SITES_FILE = "newly_discovered_sites.txt"
NEWS_SITES_CSV_FILE = "news_sites.csv"


def save_sites_to_file(sites, filepath=DISCOVERED_SITES_FILE):
    """Writes discovered news site URLs to a plain text file, one URL per line."""
    with open(filepath, "w", encoding="utf-8") as f:
        for site in sites:
            f.write(site["url"] + "\n")
    print(f"Saved {len(sites)} URLs to '{filepath}'")


def save_sites_to_csv(national_sites, state_sites, location_info, filepath=NEWS_SITES_CSV_FILE):
    """
    Saves all discovered sites to a CSV with columns:
    name, url, address, scope (national/state), country, state.
    """
    fieldnames = ["name", "url", "address", "scope", "country", "state"]
    rows = [
        {
            "name": s["name"],
            "url": s["url"],
            "address": s.get("address") or "",
            "scope": "national",
            "country": location_info["country"],
            "state": "",
        }
        for s in national_sites
    ] + [
        {
            "name": s["name"],
            "url": s["url"],
            "address": s.get("address") or "",
            "scope": "state",
            "country": location_info["country"],
            "state": location_info.get("state") or "",
        }
        for s in state_sites
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} sites ({len(national_sites)} national, {len(state_sites)} state) to '{filepath}'")


def get_location_from_coords(lat, lng, api_key):
    """
    Reverse geocodes coordinates using the Google Geocoding API.
    Returns the country and state/region (administrative_area_level_1)
    that the coordinates fall in.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lng}",
        "key": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Geocoding request error: {e}")
        return None

    results = response.json().get("results", [])
    if not results:
        print("Could not resolve a location from the provided coordinates.")
        return None

    location = {"country": None, "country_code": None, "state": None, "state_code": None}

    # Scan all result components — the state often lives in a different result
    # entry than the country-level one.
    for result in results:
        for component in result.get("address_components", []):
            types = component.get("types", [])
            if "country" in types and not location["country"]:
                location["country"] = component["long_name"]
                location["country_code"] = component["short_name"]
            elif "administrative_area_level_1" in types and not location["state"]:
                location["state"] = component["long_name"]
                location["state_code"] = component["short_name"]

    return location if location["country"] else None


def _paginated_places_search(query, api_key):
    """
    Performs a single Places Text Search query and follows nextPageToken
    until all result pages are exhausted. Returns a dict keyed by website URL
    to allow easy deduplication by the caller.
    """
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.websiteUri,places.formattedAddress,nextPageToken",
    }

    found = {}
    payload = {"textQuery": query}

    while True:
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Request error for '{query}': {e}")
            break

        data = response.json()
        for place in data.get("places", []):
            website = place.get("websiteUri")
            if website:
                found[website] = {
                    "name": place.get("displayName", {}).get("text"),
                    "url": website,
                    "address": place.get("formattedAddress"),
                }

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break
        payload = {"textQuery": query, "pageToken": next_page_token}

    return found


def find_news_sites_for_country(country_name, api_key):
    """
    Runs multiple targeted search queries for the given country to build a
    broad list of legitimate news organisation websites. Results are
    deduplicated by URL across all queries.
    """
    search_queries = [
        f"national news channel {country_name}",
        f"television news station {country_name}",
        f"national newspaper {country_name}",
        f"online news media {country_name}",
        f"radio news broadcaster {country_name}",
        f"news agency {country_name}",
    ]

    all_sites = {}
    for query in search_queries:
        print(f"  Querying: '{query}'...")
        results = _paginated_places_search(query, api_key)
        all_sites.update(results)
        print(f"  → {len(results)} sites found (total unique so far: {len(all_sites)})")

    return list(all_sites.values())

def find_news_sites_for_state(state_name, country_name, api_key):
    """
    Runs targeted search queries for state/regional news organisat  ions.
    Results are deduplicated by URL.
    """
    search_queries = [
        f"local news station {state_name} {country_name}",
        f"local newspaper {state_name}",
        f"community news {state_name} {country_name}",
        f"regional news media {state_name}",
        f"local television news {state_name}",
    ]

    all_sites = {}
    for query in search_queries:
        print(f"  Querying: '{query}'...")
        results = _paginated_places_search(query, api_key)
        all_sites.update(results)
        print(f"  \u2192 {len(results)} sites found (total unique so far: {len(all_sites)})")

    return list(all_sites.values())

# Example Execution
if __name__ == "__main__":
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Export it before running: export GOOGLE_API_KEY='your_key_here'"
        )

    # Example coordinates for Jos, Nigeria
    target_lat = 9.8965
    target_lng = 8.8583

    # ABUJA BASED MARKER
    # target_lat = 9.041802 
    # target_lng = 7.406113

    # LAGOS BASED MARKER
    # target_lat = 6.5244
    # target_lng = 3.3792

    print(f"Identifying location from coordinates ({target_lat}, {target_lng})...")
    location_info = get_location_from_coords(target_lat, target_lng, API_KEY)

    if not location_info:
        print("Failed to identify location. Exiting.")
        raise SystemExit(1)

    country = location_info["country"]
    state = location_info.get("state")

    print(f"Country : {country} ({location_info['country_code']})")
    if state:
        print(f"State   : {state} ({location_info.get('state_code', '')})")
    print()

    print(f"Searching for national news sites in {country}...")
    national_sites = find_news_sites_for_country(country, API_KEY)

    state_sites = []
    if state:
        print(f"\nSearching for state/local news sites in {state}...")
        state_sites_raw = find_news_sites_for_state(state, country, API_KEY)
        # Drop any URL already captured at the national level
        national_urls = {s["url"] for s in national_sites}
        state_sites = [s for s in state_sites_raw if s["url"] not in national_urls]

    all_sites = national_sites + state_sites
    save_sites_to_file(all_sites)
    save_sites_to_csv(national_sites, state_sites, location_info)

    print(f"\n--- National News Sites in {country} ({len(national_sites)}) ---")
    for i, site in enumerate(national_sites, 1):
        print(f"{i}. {site['name']}")
        print(f"   Website: {site['url']}")
        if site["address"]:
            print(f"   Address: {site['address']}")
        print()

    if state_sites:
        print(f"--- State/Local News Sites in {state} ({len(state_sites)}) ---")
        for i, site in enumerate(state_sites, 1):
            print(f"{i}. {site['name']}")
            print(f"   Website: {site['url']}")
            if site["address"]:
                print(f"   Address: {site['address']}")
            print()