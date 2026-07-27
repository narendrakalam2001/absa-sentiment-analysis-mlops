# ============================================================
# REVIEW SIMULATOR — Aspect-Based Sentiment Analysis (ABSA)
# 3 Scenarios: realistic_mix | negative_spike | positive_wave
# ============================================================

import requests
import random
import time
import os

# ── API URL ────────────────────────────────────────────────────
API_URL = os.getenv(
    "ABSA_API_URL",
    "http://localhost:8000"
) + "/predict"


# ============================================================
# REVIEW TEMPLATES
# ============================================================

RESTAURANT_REVIEWS = {
    "positive": [
        ("The pasta was absolutely incredible — perfectly cooked.",   "pasta"),
        ("Service was very attentive and friendly throughout.",       "service"),
        ("Ambiance was romantic and cozy, loved the lighting.",       "ambiance"),
        ("The prices are very reasonable for the quality you get.",   "price"),
        ("Food arrived hot and fresh, presentation was beautiful.",   "food"),
        ("The waiter was extremely helpful and patient.",             "waiter"),
        ("Best pizza I've ever had — thin crust, perfect toppings.", "pizza"),
        ("The dessert was divine, highly recommend the tiramisu.",    "dessert"),
    ],
    "negative": [
        ("The service was extremely slow, waited 45 minutes.",        "service"),
        ("Food was cold and tasteless when it arrived.",              "food"),
        ("The place was very noisy and we could not hear each other.", "noise"),
        ("Prices are way too high for such mediocre quality.",        "price"),
        ("The pasta was overcooked and completely bland.",            "pasta"),
        ("Staff was rude and dismissive when we asked questions.",    "staff"),
        ("The restrooms were dirty and not maintained at all.",       "cleanliness"),
        ("We had to wait over an hour for our food to arrive.",       "wait time"),
    ],
    "neutral": [
        ("The restaurant is located near the city centre.",           "location"),
        ("They have a standard menu with Italian and Indian options.", "menu"),
        ("The portion sizes are average for this price range.",       "portion size"),
        ("We made a reservation for 8pm and were seated on time.",    "reservation"),
    ],
}

LAPTOP_REVIEWS = {
    "positive": [
        ("Battery life is exceptional — lasts all day easily.",       "battery life"),
        ("The display is stunning, very sharp and bright.",           "display"),
        ("Performance is blazing fast, no lag at all.",               "performance"),
        ("Build quality feels premium, solid aluminium chassis.",     "build quality"),
        ("The keyboard is very comfortable for long typing sessions.", "keyboard"),
        ("Fan noise is barely audible even under heavy load.",        "fan noise"),
        ("Boots up in seconds, storage speed is incredible.",         "storage"),
        ("Trackpad is smooth and very responsive.",                   "trackpad"),
    ],
    "negative": [
        ("Battery drains in under 3 hours — very disappointing.",    "battery"),
        ("The screen has noticeable backlight bleeding at corners.",   "screen"),
        ("Overheats badly when running multiple applications.",       "heating"),
        ("Build feels cheap and creaky, lots of flex in the lid.",    "build quality"),
        ("The keyboard keys are mushy and hard to type accurately.",  "keyboard"),
        ("Fan is extremely loud even during basic browsing.",         "fan"),
        ("Price is too high considering the weak GPU performance.",   "price"),
        ("Customer support was unhelpful and took weeks to respond.", "support"),
    ],
    "neutral": [
        ("The laptop comes with Windows 11 pre-installed.",           "software"),
        ("Weighs around 1.8kg which is standard for this category.",  "weight"),
        ("Has 2 USB-A ports, 1 USB-C, and an HDMI port.",            "ports"),
    ],
}

BANKING_REVIEWS = {
    "positive": [
        ("The mobile app is very intuitive and easy to navigate.",    "mobile app"),
        ("Loan processing was smooth and completed within 3 days.",   "loan process"),
        ("Customer care resolved my issue in under 10 minutes.",      "customer service"),
        ("Interest rates are competitive compared to other banks.",   "interest rates"),
        ("The credit card offers excellent cashback rewards.",        "credit card"),
    ],
    "negative": [
        ("The app crashes frequently and transactions fail often.",   "mobile app"),
        ("Branch staff was rude and kept us waiting for 2 hours.",    "branch service"),
        ("Hidden charges were applied without any prior notice.",     "charges"),
        ("Loan was rejected without any clear explanation given.",    "loan"),
        ("Funds transfer takes 2-3 business days — very slow.",       "transfer speed"),
        ("ATM was out of cash for 3 consecutive days.",               "ATM"),
    ],
    "neutral": [
        ("The bank has branches in most major cities in India.",      "branch network"),
        ("Standard KYC process required for account opening.",       "KYC process"),
    ],
}


# ============================================================
# GENERATE REVIEW
# ============================================================

def generate_review(scenario: str = "realistic_mix") -> dict:
    """
    Generates a synthetic review dict for API submission.

    Scenarios:
        realistic_mix   — mixed sentiments across all domains
        negative_spike  — 70% negative reviews (stress test alerts)
        positive_wave   — 70% positive reviews (normal operations)
    """

    if scenario == "negative_spike":
        # Heavily negative — triggers alert
        sentiment = random.choices(
            ["negative", "positive", "neutral"],
            weights=[70, 20, 10]
        )[0]
        domain_reviews = random.choice([RESTAURANT_REVIEWS, LAPTOP_REVIEWS, BANKING_REVIEWS])
        domain = _get_domain(domain_reviews)

    elif scenario == "positive_wave":
        # Mostly positive — business doing well
        sentiment = random.choices(
            ["positive", "negative", "neutral"],
            weights=[70, 15, 15]
        )[0]
        domain_reviews = random.choice([RESTAURANT_REVIEWS, LAPTOP_REVIEWS])
        domain = _get_domain(domain_reviews)

    else:  # realistic_mix
        sentiment = random.choices(
            ["positive", "negative", "neutral"],
            weights=[45, 35, 20]
        )[0]
        domain_reviews = random.choice([
            RESTAURANT_REVIEWS, LAPTOP_REVIEWS, BANKING_REVIEWS
        ])
        domain = _get_domain(domain_reviews)

    options = domain_reviews.get(sentiment, domain_reviews["neutral"])
    text, aspect = random.choice(options)

    return {
        "text":        text,
        "aspect_term": aspect,
        "domain":      domain,
        "_scenario":   scenario,
        "_true_sentiment": sentiment,
    }


def _get_domain(domain_reviews: dict) -> str:
    if domain_reviews is RESTAURANT_REVIEWS:
        return "restaurants"
    elif domain_reviews is LAPTOP_REVIEWS:
        return "laptops"
    return "banking"


# ============================================================
# SEND TO API
# ============================================================

def send_review(review: dict, idx: int):
    """Sends one review to the ABSA API and prints result."""
    payload = {
        "text":        review["text"],
        "aspect_term": review["aspect_term"],
        "domain":      review["domain"],
    }
    true_sent = review.get("_true_sentiment", "?")

    try:
        response = requests.post(API_URL, json=payload, timeout=15)

        if response.status_code == 200:
            result = response.json()
            pred   = result.get("predicted_sentiment", "?")
            dec    = result.get("decision", "?")
            conf   = result.get("confidence", 0)
            match  = "✅" if pred == true_sent else "❌"

            print(
                f"[{idx+1:>3}] {match} "
                f"aspect='{review['aspect_term']:<20}' "
                f"true={true_sent:<10} "
                f"pred={pred:<10} "
                f"decision={dec:<12} "
                f"conf={conf:.3f}"
            )
        else:
            print(f"[{idx+1:>3}] ⚠️  API error: {response.status_code}")

    except Exception as e:
        print(f"[{idx+1:>3}] 🔌 Connection error: {e}")


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_reviews(n: int = 20, scenario: str = "realistic_mix", delay: float = 0.5):
    """
    Runs a full simulation sending n reviews to the ABSA API.

    Args:
        n        : number of reviews to send
        scenario : 'realistic_mix' | 'negative_spike' | 'positive_wave'
        delay    : seconds between requests
    """
    print(f"\n{'='*70}")
    print(f"ABSA Simulation | n={n} | scenario={scenario}")
    print(f"API: {API_URL}")
    print(f"{'='*70}")

    correct = 0
    total   = 0

    for i in range(n):
        review = generate_review(scenario)
        send_review(review, i)
        total += 1
        time.sleep(delay)

    print(f"{'='*70}")
    print(f"Simulation complete | {total} reviews sent")
    print(f"{'='*70}\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ABSA Review Simulator")
    parser.add_argument("--n",        type=int,   default=20,               help="Number of reviews")
    parser.add_argument("--scenario", type=str,   default="realistic_mix",
                        choices=["realistic_mix", "negative_spike", "positive_wave"])
    parser.add_argument("--delay",    type=float, default=0.5,              help="Delay between requests (s)")
    args = parser.parse_args()

    simulate_reviews(n=args.n, scenario=args.scenario, delay=args.delay)
