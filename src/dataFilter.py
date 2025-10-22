import pandas as pd

crime_mapping = {
    # 🟥 Rape-related
    "RAPE": "Rape-related",
    "Rape (Section 376 IPC)": "Rape-related",
    "Custodial Rape": "Rape-related",
    "Custodial_Gang Rape": "Rape-related",
    "Custodial_Other Rape": "Rape-related",
    "Rape other than Custodial": "Rape-related",
    "Rape_Gang Rape": "Rape-related",
    "Rape_Others": "Rape-related",
    "Attempt to commit Rape (Section 376 & 511 IPC)": "Rape-related",
    "ATTEMPT TO RAPE": "Rape-related",

    # 🟨 Kidnapping & Abduction
    "KIDNAPPING": "Kidnapping & Abduction",
    "ABDUCTION": "Kidnapping & Abduction",
    "Kidnapping & Abduction of Women (Section 363,364,364A, 366-369 IPC)": "Kidnapping & Abduction",
    "Kidnaping & Abduction Section 363 IPC": "Kidnapping & Abduction",
    "Kidnaping & Abduction in order to Murder Section 364 IPC": "Kidnapping & Abduction",
    "Kidnapping for Ransom Section 364A": "Kidnapping & Abduction",
    "Kidnapping & Abduction of Women to compel her for marriage": "Kidnapping & Abduction",
    "Kidnapping & Abduction of Women-Other": "Kidnapping & Abduction",

    # 🟩 Dowry-related
    "DOWRY": "Dowry-related",
    "Dowry Deaths (Section 304-B IPC)": "Dowry-related",
    "DOWRY DEATH": "Dowry-related",
    "Dowry Deaths": "Dowry-related",
    "10 - Dowry Prohibition Act, 1961": "Dowry-related",
    "Dowry Prohibition Act": "Dowry-related",

    # 🟦 Assault & Modesty-related
    "ASSAULT": "Assault & Modesty-related",
    "SEXUAL HARASSMENT": "Assault & Modesty-related",
    "STALKING": "Assault & Modesty-related",
    "VOYEURISM": "Assault & Modesty-related",
    "INSULT": "Assault & Modesty-related",
    "OUTRAGE HER MODESTY": "Assault & Modesty-related",
    "Assault on Women with intent to outrage her Modesty_Total (Section 354 IPC)": "Assault & Modesty-related",
    "Sexual Harassment": "Assault & Modesty-related",
    "Assault on women with intent to Disrobe": "Assault & Modesty-related",
    "Voyeurism": "Assault & Modesty-related",
    "Stalking": "Assault & Modesty-related",
    "Others": "Assault & Modesty-related",
    "ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY": "Assault & Modesty-related",
    "Assault on women with intent to outrage her Modest": "Assault & Modesty-related",
    "Insult to the Modesty of Women_Total (Section 509 IPC)": "Assault & Modesty-related",
    "At Office premises": "Assault & Modesty-related",
    "Other places related to work": "Assault & Modesty-related",
    "In Public Transport system": "Assault & Modesty-related",
    "in Other Places": "Assault & Modesty-related",

    # 🟪 Domestic & Cruelty
    "CRUELTY": "Domestic & Cruelty",
    "HUSBAND": "Domestic & Cruelty",
    "SUICIDE": "Domestic & Cruelty",
    "Cruelty by Husband or his Relatives (Section 498A)": "Domestic & Cruelty",
    "Cruelty by Husband or his relatives": "Domestic & Cruelty",
    "CRUELTY BY HUSBAND OR RELATIVES": "Domestic & Cruelty",
    "Abetment of Suicides of Women u/s 306 IPC": "Domestic & Cruelty",

    # 🟧 Other Acts
    "IMPORTATION": "Other Acts",
    "INDECENT REPRESENTATION": "Other Acts",
    "SATI": "Other Acts",
    "IMMORAL TRAFFIC": "Other Acts",
    "DOMESTIC VIOLENCE": "Other Acts",
    "ITP": "Other Acts",
    "Total Crimes against Women": "Other Acts",
    "Importation of Girls from Foreign Country (Section 366B IPC)": "Other Acts",
    "Indecent Representation of Women (Prohibition) Act, 1986": "Other Acts",
    "INDECENT REPRESENTATION OF WOMEN(PREVENTION)ACT": "Other Acts",
    "Commission of Sati Prevention Act 1987": "Other Acts",
    "Commission of Sati (P) Act": "Other Acts",
    "Protection of Women from Domestic Violence Act, 2005": "Other Acts",
    "Immoral Traffic (Prevention) Act (Women Cases only)": "Other Acts",
    "Under ITP Section 5": "Other Acts",
    "Under ITP Section 6": "Other Acts",
    "Under ITP Section 7": "Other Acts",
    "Under ITP Section 8": "Other Acts",
    "Other Sections under ITP Act": "Other Acts",
    "IMMORAL TRAFFIC(PREVENTION)ACT": "Other Acts",
}

def classify_crime(crime_name):
    for key, category in crime_mapping.items():
        if key.lower() in crime_name.lower():
            return category
    return "Other/Unclassified"

crime_data = pd.read_csv("Crime-Prediction/merged_crime_data.csv")

crime_data["Main Category"] = crime_data["Crime Head"].apply(classify_crime)
crime_data.to_csv("Crime-Prediction/classified_crime_data.csv", index=False)