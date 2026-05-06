# NFL Fourth Down Decision Dashboard

An interactive NFL analytics dashboard that models fourth-down decision making using expected points added (EPA), conversion probability, and historical play-by-play data.

Built with:
- Streamlit
- Plotly
- Pandas
- nflreadpy

The dashboard evaluates:
- Go for it
- Punt
- Field goal

using a true EPA-based decision framework.

---

## Features

### True Fourth-Down Decision Model

Instead of simply comparing “go” vs “don’t go,” this dashboard estimates:

- Expected EPA from going for it
- Expected EPA from punting
- Expected EPA from attempting a field goal

Then recommends the highest-value decision.

---

### Expected Conversion Probability Model

Conversion probabilities are estimated from historical NFL fourth-down situations using:

- Distance to go
- Field position
- Game situation

The model uses smoothing to reduce small-sample noise.

---

### EPA Over Expected

Measure how teams perform relative to expectation.

Metrics include:

- Conversion Over Expected (COE)
- EPA Over Expected (EPAOE)
- Expected conversion value
- Decision edge

---

### Interactive Visualizations

Includes:

- Go/Punt/FG EPA comparison charts
- Team comparison mode
- Conversion rate vs expected conversion rate
- EPA over expected by field position
- Fourth-down decision maps
- Recommendation mix by situation

---

## Example Dashboard Questions

- Should teams go for it more often on 4th-and-short?
- Which NFL teams outperform expected conversion rates?
- When is punting actually optimal?
- How aggressive are elite offenses?
- Which teams gain the most EPA on fourth downs?

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/fourth-down-dashboard.git
cd fourth-down-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run dashboard.py
```

---

## Requirements

Create a `requirements.txt` file containing:

```txt
streamlit
pandas
plotly
nflreadpy
```

---

## Data Source

NFL play-by-play data is loaded using:

- nflreadpy

Seasons included:

- 2023
- 2024
- 2025

---

## Dashboard Metrics

| Metric | Description |
|---|---|
| Conversion Rate | Actual fourth-down conversion rate |
| Expected Conversion Rate | Model-estimated conversion probability |
| Conversion Over Expected | Actual minus expected conversion rate |
| EPA | Expected Points Added |
| EPA Over Expected | Actual EPA minus expected EPA |
| Go EPA | Estimated EPA from going for it |
| Punt EPA | Estimated EPA from punting |
| FG EPA | Estimated EPA from attempting a field goal |
| Decision Edge | Difference between best and second-best decision |

---

## Project Structure

```text
fourth-down-dashboard/
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## Streamlit Cloud Deployment

1. Push repository to GitHub
2. Go to Streamlit Community Cloud
3. Create a new app
4. Select your repository
5. Set:

```text
Main file path: dashboard.py
```

6. Deploy

---

## Future Improvements

Potential upgrades:

- Win probability integration
- Live game decision engine
- Expected field goal success model
- Machine learning conversion model
- Coach aggressiveness profiles
- Drive outcome simulations
- Play-type recommendation engine
- Team tendency modeling

---

## Analytics Notes

This project separates:

- Offensive go-for-it attempts
- Punts
- Field goals

into independent EPA models.

This avoids the common mistake of calculating conversion rates using all fourth downs, which artificially depresses expected conversion probabilities.
