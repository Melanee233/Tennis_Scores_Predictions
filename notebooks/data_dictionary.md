## Data Dictionary — WTA Matches Dataset

This dataset contains historical match-level data for professional women’s tennis (WTA). Each row represents a single match and includes information about the tournament, match context, players (winner and loser), rankings and detailed match statistics.

### Tournament Information

* **tourney_id** — Unique tournament identifier (e.g., 2020-888). The first four digits indicate the year.
* **tourney_name** — Name of the tournament.
* **surface** — Court surface (Hard, Clay, Grass, Carpet).
* **draw_size** — Number of players in the tournament draw (often rounded to a power of two).
* **tourney_level** — Tournament category (e.g., G = Grand Slam, P = Premier, I = International, etc.).
* **tourney_date** — Tournament start date in YYYYMMDD format (typically the Monday of tournament week).

### Match Information

* **match_num** — Match identifier within the tournament.
* **round** — Tournament round (e.g., F, SF, QF, R16, etc.).
* **best_of** — Maximum number of sets (typically 3 in women’s tennis).
* **minutes** — Match duration in minutes (if available).
* **score** — Final match score in set notation.

### Player Identification

For both the winner and loser:

* **winner_id / loser_id** — Unique player identifier.
* **winner_name / loser_name** — Player name.
* **winner_ioc / loser_ioc** — Three-letter country code.

### Player Attributes (Pre-Match)

* **winner_seed / loser_seed** — Tournament seeding.
* **winner_entry / loser_entry** — Entry type (e.g., WC = Wild Card, Q = Qualifier, LL = Lucky Loser, PR = Protected Ranking).
* **winner_hand / loser_hand** — Dominant hand (R = Right, L = Left, U = Unknown).
* **winner_ht / loser_ht** — Height in centimeters (if available).
* **winner_age / loser_age** — Age in years at the time of the tournament.
* **winner_rank / loser_rank** — WTA ranking at tournament start.
* **winner_rank_points / loser_rank_points** — Ranking points.

### Match Statistics (Post-Match)

Statistics recorded separately for the winner (w_) and loser (l_):

* **ace** — Number of aces.
* **df** — Double faults.
* **svpt** — Total serve points played.
* **1stIn** — First serves made.
* **1stWon** — Points won on first serve.
* **2ndWon** — Points won on second serve.
* **SvGms** — Service games played.
* **bpSaved** — Break points saved.
* **bpFaced** — Break points faced.
