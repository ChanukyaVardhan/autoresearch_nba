# What the AI Did — in plain English

## The setup (one paragraph, no jargon)
We're betting on live NBA games on a prediction market (Kalshi). Every minute, a price
moves up and down as the game unfolds — like a stock. We give the trader **$1 per game**
to work with; it watches the game + the price and decides each minute: **buy, sell, or
wait.** (So a score of +0.72 means it turned that $1 into ~$1.72 on average.) Then we
gave a *second* AI (the "researcher") one job: **keep rewriting the trader's code to make
more money** — automatically, no human helping. It tries an idea, we test it, and we
only keep it if it actually earns more. Money made per game is the score.

---

## What happened, iteration by iteration

**Start (baseline):** the trader already makes money but is hyperactive — it trades
**72 times a game**, flipping in and out constantly. Profitable but messy.

**Round 1 — "stop trading so much."** The AI noticed the over-trading and added a
penalty for piling on extra bets. Result: trades dropped to ~38 and **profit nearly
doubled.** Lesson: *trade smarter, not more.*

**Round 2 — "look at the real price you'd pay."** It added features for the gap between
"what we think will happen" and the *actual buy price and sell price* (not the average).
This makes the trader honest about the real cost of trading. Cleaner, slightly better.

**Rounds 3–5 — a slump.** Three ideas in a row didn't beat the best, so they were all
**thrown away.** One of them even made the trader freeze up and never trade (0 profit) —
the system caught it instantly and discarded it. *Most experiments fail; that's normal.*

**Round 6 — the breakthrough.** It added "momentum" features — not just *is* there an
edge, but *is the edge growing or shrinking right now*. This let the trader time its
moves better. **New best profit — and it even did better on games it had never seen.**
Bonus: its first attempt this round had a bug; the system handed it the error and it
**fixed its own code** and tried again.

**The score over time:** 0.37 → 0.66 → 0.67 → (slump) → **0.72** — about **double** where
it started, with **zero human edits** and roughly **$2 of AI cost.**

---

## What the trader actually "looks at" (the 62 features)

Every minute, the trader sees a snapshot of 62 numbers. In plain terms:

**The price & how it's moving (the market)**
- current price / implied win-chance, the buy-sell gap (`spread`)
- how fast the price is moving (1, 3, 5 minutes ago) and accelerating
- trading volume and surges in it, open interest changes

**What's happening in the game**
- score margin, which quarter, time left in quarter / game
- recent scoring runs (last 60s, 180s), who has the ball, just had a timeout

**Our "edge" — where we think the market is wrong (the most important part)**
- our model's win-probability vs the market's price = the **edge**
- edge measured at the real **buy price** and **sell price** (`buy_edge`, `sell_edge`)
- edge *after subtracting trading cost* (`net_buy_edge`, `net_hold_edge`)
- **how the edge is changing** over the last 3 min (`edge_delta_180`) — momentum
- how far the market price has drifted from what the score implies

**Our own position (so it knows what it's holding)**
- are we holding?, entry price, current profit/loss, how long we've held, budget left

**The players on the floor**
- top 3 players per team: are they in, their recent scoring/rebound/assist rate, fouls
  — all reconstructed minute-by-minute from play-by-play (never peeking at the final box score)

*(The AI invented the bold ones — `buy_edge`, `sell_edge`, `net_*_edge`, `edge_delta_180`,
the momentum features — on its own across the rounds. The plain price/score/player
features were the starting set.)*

**One honest note:** the profit number is a bit optimistic — our cost model is generous
about trading fees. The *trend* (it keeps improving) and the fact it works on unseen
games are the real, trustworthy parts.
