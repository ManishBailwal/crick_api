import requests

API_KEY = "17eb0cc2-ddf1-4019-ba2f-073276171df8"
url = f"https://api.cricapi.com/v1/cricRankings?apikey={API_KEY}"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    if data.get("status") == "success":
        print("🎉 API key is working! Here's a sample of the data:")
        for entry in data["data"]:
            if entry["format"] == "odi" and entry["type"] == "team":
                for team in entry["ranking"][:5]:  # top 5 teams
                    print(f"{team['team']} - Rank: {team['rank']}")
    else:
        print("🚫 API responded but status is not success:")
        print(data)
else:
    print(f"❌ Failed to connect. Status Code: {response.status_code}")
