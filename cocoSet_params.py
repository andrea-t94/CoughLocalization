"""COCO dataset parameters about info, licenses, and categories"""

from datetime import datetime
now = datetime.datetime.now()

#general informations
    info = {
        "description": "VOICEMED Cough Dataset",
        "url": "",
        "version": "1.0.0",
        "year": now.year,
        "contributor": "Voicemed ML Team",
        "date_created": date.today()
    }

#eventual licenses
    licenses = [
        {
            "url": "https://www.voicemed.io/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        }
    ]

#categories for sound localization
    categories = [
        {"supercategory": "human_sound", "id": 0, "name": "cough"},
        {"supercategory": "human_sound", "id": 1, "name": "breath"},
        {"supercategory": "human_sound", "id": 2, "name": "speech"},
        {"supercategory": "other", "id": 3, "name": "other"}
    ]
