"""
dataset_generator.py — Large-Scale Synthetic Dataset Generation
Domain: Big Data Analytics
Generates a massive dataset (50,000+ samples) using combinatorial templates and vast vocabulary
to ensure the model learns robust, varied representations of framing bias.
"""

import pandas as pd
import random

# Massive vocabulary banks for each frame
vocab = {
    "Economic": {
        "subjects": ["The economy", "Inflation", "The stock market", "Global trade", "Unemployment", "Corporate profit", "Taxation", "The banking sector", "Consumer spending", "The housing market", "Interest rates", "National debt", "The GDP", "Income inequality", "Minimum wage"],
        "verbs": ["is impacting", "has disrupted", "will transform", "is driving up", "has destabilized", "continues to affect", "is reshaping", "threatens", "boosts", "undermines", "is fluctuating due to", "has recovered from", "is heavily dependent on"],
        "objects": ["market stability.", "prices across the board.", "investment portfolios.", "the cost of living.", "small business growth.", "international supply chains.", "consumer confidence.", "economic recovery.", "the workforce.", "retirement funds.", "foreign exchange rates."],
        "adjectives": ["severe", "unprecedented", "sluggish", "rapid", "volatile", "bullish", "bearish", "stagnant", "robust", "fragile"],
        "contexts": ["amid rising interest rates.", "during this fiscal quarter.", "in the global market.", "according to economists.", "despite government intervention.", "following the market crash."]
    },
    "Political": {
        "subjects": ["The government", "The senate", "Parliament", "The opposition party", "The president", "Local elections", "New legislation", "Voter turnout", "The supreme court", "Democratic institutions", "Diplomatic relations", "The prime minister", "Campaign strategies"],
        "verbs": ["has passed", "is debating", "has vetoed", "proposed", "is investigating", "has undermined", "will vote on", "is negotiating", "has reformed", "is challenging", "faces scrutiny over", "has enacted", "is pivoting on"],
        "objects": ["the new tax bill.", "foreign policy initiatives.", "healthcare reform.", "election integrity laws.", "the annual budget.", "diplomatic treaties.", "civil rights legislation.", "political funding transparency.", "gerrymandering boundaries.", "campaign finance."],
        "adjectives": ["controversial", "bipartisan", "radical", "conservative", "progressive", "historic", "polarizing", "sweeping", "unconstitutional", "unpopular"],
        "contexts": ["ahead of the midterms.", "in a surprising move.", "after weeks of deadlock.", "sparking public outrage.", "with across-the-aisle support.", "causing a political crisis."]
    },
    "Social": {
        "subjects": ["Civil rights groups", "The local community", "Public schools", "Healthcare workers", "Marginalized demographics", "Social media movements", "Youth organizations", "The healthcare system", "Cultural institutions", "Activists", "Public welfare", "Minority communities"],
        "verbs": ["are demanding", "have organized for", "are advocating for", "are protesting against", "highlight the need for", "are fighting for", "have raised awareness about", "are disproportionately affected by", "are challenging", "are defending"],
        "objects": ["racial equality.", "better access to healthcare.", "education reform.", "gender parity.", "affordable housing.", "human rights.", "social justice.", "community support systems.", "systemic discrimination.", "wealth redistribution.", "public health initiatives."],
        "adjectives": ["systemic", "widespread", "deep-rooted", "cultural", "societal", "marginalized", "vulnerable", "equitable", "inclusive", "institutional"],
        "contexts": ["in marginalized neighborhoods.", "across the nation.", "on college campuses.", "in recent protests.", "due to historical inequities.", "in the wake of the tragedy."]
    },
    "Security": {
        "subjects": ["Military forces", "The defense department", "Border patrol", "Counter-terrorism units", "Global security alliances", "Armed insurgents", "The police force", "Cybersecurity experts", "Intelligence agencies", "National guard troops", "Naval fleets"],
        "verbs": ["have launched", "are preparing for", "have escalated", "intercepted", "are deploying to", "have neutralized", "are investigating", "warned of", "responded to", "have fortified", "are retreating from"],
        "objects": ["a massive offensive.", "border hostilities.", "terrorist cells.", "cyber attacks.", "enemy combatants.", "domestic uprisings.", "foreign espionage.", "drone strikes.", "nuclear proliferation threats.", "military blockades."],
        "adjectives": ["violent", "lethal", "covert", "strategic", "tactical", "dangerous", "unstable", "hostile", "armed", "defensive"],
        "contexts": ["along the disputed border.", "in the warzone.", "in a classified operation.", "following the terror threat.", "to maintain national security.", "amid rising geopolitical tensions."]
    },
    "Environment": {
        "subjects": ["Climate change", "Global warming", "Deforestation", "Ocean acidification", "Renewable energy", "Carbon emissions", "Plastic pollution", "Extreme weather events", "Wildlife conservation efforts", "The fossil fuel industry", "Greenhouse gases", "Melting glaciers"],
        "verbs": ["is exacerbating", "has ruined", "is threatening", "will destroy", "is transforming", "have damaged", "is accelerating", "is harming", "polluted", "has devastated", "is improving"],
        "objects": ["global biodiversity.", "coastal cities.", "air quality.", "the ozone layer.", "natural habitats.", "endangered species.", "global temperatures.", "agricultural yields.", "marine ecosystems.", "drinking water supplies.", "the Amazon rainforest."],
        "adjectives": ["catastrophic", "devastating", "sustainable", "toxic", "eco-friendly", "hazardous", "irreversible", "polluted", "reusable", "unprecedented"],
        "contexts": ["due to rising carbon levels.", "according to climate scientists.", "in the upcoming decade.", "causing mass extinctions.", "with irreversible consequences.", "despite global climate treaties."]
    }
}

# Shared vocabulary pool that all categories pull from (introduces realistic overlap)
shared_vocab = {
    "subjects": ["The new policy", "Official reports", "Public response", "Experts", "International observers", "Leading analysts", "The latest data", "Community leaders", "Research", "Media coverage"],
    "verbs": ["have indicated", "suggest that", "might lead to", "reveal that", "point towards", "are influenced by", "reflect", "demonstrate"],
    "objects": ["future developments.", "significant changes.", "widespread impact.", "long-term consequences.", "shifting trends.", "the current situation.", "public perception."],
    "noise": [
        "In a rapidly changing world,", "According to various sources,", "It remains to be seen if", "Interestingly enough,", "While some disagree,", 
        "Notably,", "On the other hand,", "In conclusion,"
    ]
}

templates = [
    "{subject} {verb} {object}",
    "{subject} {verb} the {adjective} {object}",
    "Due to {adjective} conditions, {subject} {verb} {object}",
    "{subject} {verb} {object} {context}",
    "{context} {subject} {verb} the {adjective} {object}",
    "Experts warn that {subject} {verb} {object}",
    "We are seeing that {subject} {verb} {object} {context}",
    "The reality is that the {adjective} {subject} {verb} {object}",
    "Common ground exists where {subject} {verb} {object}", # Neutral template
]

propaganda_injections = [
    " WAKE UP! The regime is hiding the truth.",
    " It's a total catastrophe and they are lying to us.",
    " The corrupt agenda is finally exposed.",
    " This is an absolute disaster engineered by the elite.",
    " Don't believe the fake news, this will destroy us."
]

def generate_dataset(num_samples=50000):
    data = []
    keys = list(vocab.keys())
    
    print(f"Generating {num_samples} samples with Realism Calibration...")
    
    for _ in range(num_samples):
        # 1. Select a random frame but allow for "fuzzy" cross-over
        primary_label = random.choice(keys)
        # 20% chance to pull vocabulary from a DIFFERENT frame while keeping the primary label (Ambiguity)
        if random.random() < 0.20:
            secondary_label = random.choice(keys)
            v = vocab[secondary_label]
        else:
            v = vocab[primary_label]
            
        # 2. Mix in Shared Vocab (30% chance per slot) to reduce keyword isolation
        subject = random.choice(shared_vocab["subjects"]) if random.random() < 0.3 else random.choice(v["subjects"])
        verb = random.choice(shared_vocab["verbs"]) if random.random() < 0.3 else random.choice(v["verbs"])
        obj = random.choice(shared_vocab["objects"]) if random.random() < 0.3 else random.choice(v["objects"])
        adj = random.choice(v["adjectives"])
        context = random.choice(v["contexts"])
        
        template = random.choice(templates)
        
        # 3. Text Assembly
        text = template.replace("{subject}", subject) \
                       .replace("{verb}", verb) \
                       .replace("{object}", obj) \
                       .replace("{adjective}", adj) \
                       .replace("{context}", context)
        
        # 4. Add Ambient Noise (15% chance at start/end)
        if random.random() < 0.15:
            noise = random.choice(shared_vocab["noise"])
            text = f"{noise} {text}" if random.random() < 0.5 else f"{text} {noise}"

        # Capitalize and clean
        text = text[0].upper() + text[1:]
        if not text.endswith('.'): text += '.'
        
        # Remove any occurrence of the label name itself (Explicit Cheat Prevention)
        for label_name in keys:
            text = text.replace(label_name, "the industry")
            text = text.replace(label_name.lower(), "the sector")

        # 5. 10% Propaganda Injection
        if random.random() < 0.1:
            text += random.choice(propaganda_injections)
            
        data.append({"text": text, "label": primary_label})
        
    random.shuffle(data)
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset(55000)
    df.to_csv("dataset.csv", index=False)
    print(f"Generated {len(df)} realistic samples and saved to dataset.csv")
    print("\nDataset Distribution:")
    print(df["label"].value_counts())
