import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load data
df = pd.read_csv("odi_Matches_Data.csv")
df = df[['Team1 Name', 'Team2 Name', 'Toss Winner', 'Toss Winner Choice', 'Match Winner', 'Match Venue (Stadium)']].dropna()
df.columns = ['team1', 'team2', 'toss_winner', 'toss_decision', 'winner', 'venue']

# Filter teams with enough data
team_counts = df['winner'].value_counts()
valid_teams = team_counts[team_counts >= 30].index
df = df[df['winner'].isin(valid_teams)].copy()

# Load ICC rankings
rankings = pd.read_csv("icc_odi_rankings.csv")
rankings.columns = ['rank', 'team', 'matches', 'points', 'rating']
rankings['team'] = rankings['team'].str.strip()
rank_map = dict(zip(rankings['team'], rankings['rank']))
df['team1_rank'] = df['team1'].map(rank_map)
df['team2_rank'] = df['team2'].map(rank_map)
df.dropna(subset=['team1_rank', 'team2_rank'], inplace=True)
df['rank_diff'] = df['team1_rank'] - df['team2_rank']

# Normalize toss decision
df['toss_decision'] = df['toss_decision'].str.lower().replace({'bowl': 'field'})

# 🔥 Global Head-to-Head Win Difference
h2h_win_diff = {}
for index, row in df.iterrows():
    key = tuple(sorted([row['team1'], row['team2']]))
    if key not in h2h_win_diff:
        matches = df[((df['team1'] == key[0]) & (df['team2'] == key[1])) | ((df['team1'] == key[1]) & (df['team2'] == key[0]))]
        t1_wins = sum(matches['winner'] == key[0])
        t2_wins = sum(matches['winner'] == key[1])
        h2h_win_diff[key] = t1_wins - t2_wins
    diff = h2h_win_diff[key]
    df.at[index, 'h2h_diff'] = diff if row['team1'] == key[0] else -diff

# 🔥 Venue-Specific Head-to-Head Win Difference
venue_h2h_diff = {}
for index, row in df.iterrows():
    venue = row['venue']
    key = (venue, *sorted([row['team1'], row['team2']]))
    if key not in venue_h2h_diff:
        matches = df[(df['venue'] == venue) &
                     (((df['team1'] == key[1]) & (df['team2'] == key[2])) |
                      ((df['team1'] == key[2]) & (df['team2'] == key[1])))]
        team1_wins = sum(matches['winner'] == key[1])
        team2_wins = sum(matches['winner'] == key[2])
        venue_h2h_diff[key] = team1_wins - team2_wins
    diff = venue_h2h_diff[key]
    df.at[index, 'venue_h2h_diff'] = diff if row['team1'] == key[1] else -diff

# Encode categorical variables
le_team = LabelEncoder()
all_teams = pd.concat([df['team1'], df['team2'], df['toss_winner']]).unique()
le_team.fit(all_teams)
df['team1_enc'] = le_team.transform(df['team1'])
df['team2_enc'] = le_team.transform(df['team2'])
df['toss_winner_enc'] = le_team.transform(df['toss_winner'])

le_decision = LabelEncoder()
le_decision.fit(['bat', 'field'])
df['toss_decision_enc'] = le_decision.transform(df['toss_decision'])

le_venue = LabelEncoder()
df['venue_enc'] = le_venue.fit_transform(df['venue'])

le_winner = LabelEncoder()
df['winner_enc'] = le_winner.fit_transform(df['winner'])

# Final feature matrix
X = df[['team1_enc', 'team2_enc', 'toss_winner_enc', 'toss_decision_enc',
        'venue_enc', 'team1_rank', 'team2_rank', 'rank_diff', 'h2h_diff', 'venue_h2h_diff']]
y = df['winner_enc']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"📊 XGBoost Accuracy on Test Set: {accuracy * 100:.2f}%")

# Save model and encoders
joblib.dump(model, 'winner_predictor_model.pkl')
joblib.dump(le_team, 'team_encoder.pkl')
joblib.dump(le_decision, 'decision_encoder.pkl')
joblib.dump(le_venue, 'venue_encoder.pkl')
joblib.dump(le_winner, 'winner_encoder.pkl')

print("✅ Model and encoders with venue-specific H2H saved successfully.")
