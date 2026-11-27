import asyncio
import uuid
import random
import pandas as pd
from datetime import datetime

from sqlalchemy import text

from infrastructure.db.model.engine import get_session_ctx

# ================= CONFIG =================
LO_MIN = 20
LO_MAX = 100

EXPORT_MODE = "csv"
# "sql" | "csv" | "both"

CSV_LO_FILE = "lo.csv"
CSV_LO_KP_FILE = "lo_kp.csv"

DIFF_MIN = 0.1
DIFF_MAX = 1.0
DYNAMIC_MODE='noise'
DYNAMIC_NOISE_RANGE = 0.15

WEIGHT_MIN = 0.1
WEIGHT_MAX = 1.0

RATING_MIN = 0.5
RATING_MAX = 5.0
# ==========================================


LO_TYPE_MAPPING = {
    # Animation thuần thị giác
    "animation_visual": {
        "visual_verbal": 0.9,  # Thuần visual
        "active_reflective": 0.3,
        "sensing_intuitive": 0.6,
        "sequential_global": 0.7,
        "is_graded": False
    },

    # Animation dạng video (cân bằng visual và verbal)
    "animation_video": {
        "visual_verbal": 0.1,  # Cân bằng giữa visual và verbal
        "active_reflective": 0.4,
        "sensing_intuitive": 0.5,
        "sequential_global": 0.6,
        "is_graded": False
    },

    # Reading material chỉ có chữ
    "reading_material_text": {
        "visual_verbal": -0.8,  # Nghiêng về verbal
        "active_reflective": 0.8,
        "sensing_intuitive": 0.3,
        "sequential_global": -0.8,
        "is_graded": False
    },

    # Reading material có chữ và hình ảnh
    "reading_material_text_image": {
        "visual_verbal": 0.3,  # Cân bằng một chút giữa visual và verbal
        "active_reflective": 0.7,
        "sensing_intuitive": 0.4,
        "sequential_global": -0.6,
        "is_graded": False
    },

    "exercise": {
        "visual_verbal": 0.0,
        "active_reflective": -0.9,
        "sensing_intuitive": -0.8,
        "sequential_global": -0.7,
        "is_graded": True
    },

    "reflection_quiz": {
        "visual_verbal": 0.4,
        "active_reflective": 0.9,
        "sensing_intuitive": 0.4,
        "sequential_global": -0.6,
        "is_graded": True
    },

    "self_assessment_test": {
        "visual_verbal": 0.2,
        "active_reflective": 0.8,
        "sensing_intuitive": -0.3,
        "sequential_global": -0.8,
        "is_graded": True
    },

    "discussion_forum": {
        "visual_verbal": 0.8,
        "active_reflective": -0.8,
        "sensing_intuitive": 0.6,
        "sequential_global": 0.8,
        "is_graded": False
    },

    "real_life_application": {
        "visual_verbal": -0.5,
        "active_reflective": -0.9,
        "sensing_intuitive": -0.9,
        "sequential_global": 0.6,
        "is_graded": False
    }
}

LO_TYPE_DISTRIBUTION = {
    "reading_material_text": 0.15,
    "reading_material_text_image": 0.15,
    "exercise": 0.25,
    "animation_video": 0.15,
    "animation_visual": 0.05,
    "reflection_quiz": 0.10,
    "self_assessment_test": 0.10,
    "discussion_forum": 0.05,
    "real_life_application": 0.10
}

def sample_lo_type():
    types = list(LO_TYPE_DISTRIBUTION.keys())
    weights = list(LO_TYPE_DISTRIBUTION.values())
    return random.choices(types, weights=weights, k=1)[0]

def mapping_lo_type_to_real_lo_type(lo_type):
    if lo_type == "animation_video" or lo_type == "animation_visual":
        return "animation"
    elif lo_type == "reading_material_text" or lo_type == "reading_material_text_image":
        return "reading_material"
    return lo_type

def add_noise(v):
    noisy = v + random.uniform(-0.05, 0.05)
    return round(min(1.0, max(-1.0, noisy)), 3)


def generate_weight():
    return round(random.uniform(WEIGHT_MIN, WEIGHT_MAX), 3)

def generate_difficulty_init():
    return round(random.uniform(DIFF_MIN, DIFF_MAX), 3)


def generate_difficulty_dyn(init_value):
    if DYNAMIC_MODE == "same":
        return init_value

    if DYNAMIC_MODE == "noise":
        noisy = init_value + random.uniform(-DYNAMIC_NOISE_RANGE, DYNAMIC_NOISE_RANGE)
        return round(min(1.0, max(0.1, noisy)), 3)

def generate_rating():
    """Generate rating in numeric(3,2) format: total 3 digits with 2 decimal places"""
    return round(random.uniform(RATING_MIN, RATING_MAX), 2)

def generate_lo_for_kp(kp):
    lo_rows = []
    lo_kp_rows = []

    lo_count = random.randint(LO_MIN, LO_MAX)

    for i in range(lo_count):
        lo_type = sample_lo_type()
        style = LO_TYPE_MAPPING[lo_type]
        lo_id = str(uuid.uuid4())

        init_diff = generate_difficulty_init()

        lo_rows.append({
            "id": lo_id,
            "lo_id": kp["id"],
            "title": f"{kp['title']} - {lo_type.replace('_',' ').title()}",
            "type": mapping_lo_type_to_real_lo_type(lo_type),
            "difficulty_init": init_diff,
            "difficulty_dyn": generate_difficulty_dyn(init_diff),
            "duration_nominal": random.randint(5, 30),
            "mean_score": None,
            "mean_time": None,
            "is_graded": style["is_graded"],
            "rating": generate_rating(),
            "visual_verbal": add_noise(style["visual_verbal"]),
            "active_reflective": add_noise(style["active_reflective"]),
            "sensing_intuitive": add_noise(style["sensing_intuitive"]),
            "sequential_global": add_noise(style["sequential_global"]),
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })

        lo_kp_rows.append({
            "lo_id": lo_id,
            "kp_id": kp["id"],
            "weight": generate_weight()
        })

    return lo_rows, lo_kp_rows


def export_sql(lo_list, lo_kp_list):
    print("\n-- LO INSERT --\n")
    for lo in lo_list:
        print(f"""
INSERT INTO lo (id, lo_id, title, type,
difficulty_init, duration_nominal, is_graded, rating,
visual_verbal, active_reflective,
sensing_intuitive, sequential_global,
created_at, updated_at)
VALUES (
'{lo["id"]}',
'{lo["lo_id"]}',
'{lo["title"]}',
'{lo["type"]}',
{lo["difficulty_init"]},
{lo["duration_nominal"]},
{str(lo["is_graded"]).lower()},
{lo["rating"]},
{lo["visual_verbal"]},
{lo["active_reflective"]},
{lo["sensing_intuitive"]},
{lo["sequential_global"]},
NOW(),
NOW()
);
""")

    print("\n-- LO_KP INSERT --\n")
    for row in lo_kp_list:
        print(f"""
INSERT INTO lo_kp (lo_id, kp_id, weight)
VALUES ('{row["lo_id"]}', '{row["kp_id"]}', {row["weight"]});
""")


def export_csv(lo_list, lo_kp_list):
    df_lo = pd.DataFrame(lo_list)
    df_lo_kp = pd.DataFrame(lo_kp_list)

    df_lo.to_csv(CSV_LO_FILE, index=False)
    df_lo_kp.to_csv(CSV_LO_KP_FILE, index=False)

    print(f"CSV exported: {CSV_LO_FILE}, {CSV_LO_KP_FILE}")


# ================= EXAMPLE =================

async def generate_from_db(export_mode="csv"):
    query = text("""SELECT id, title FROM elearningapp.kp""")
    all_lo = []
    all_lo_kp = []

    async with get_session_ctx() as s:
        result = await s.execute(query)
        kp_list = result.mappings().all()
        print("Loaded KP list from DB:", len(kp_list))
        for kp in kp_list:
            lo_rows, lo_kp_rows = generate_lo_for_kp(kp)
            all_lo.extend(lo_rows)
            all_lo_kp.extend(lo_kp_rows)

        if export_mode == "sql":
            export_sql(all_lo, all_lo_kp)
        elif export_mode == "csv":
            export_csv(all_lo, all_lo_kp)
        elif export_mode == "both":
            export_sql(all_lo, all_lo_kp)
            export_csv(all_lo, all_lo_kp)

def test():
    kp_test_data = [
        {"id": str(uuid.uuid4()), "title": "SQL Joins"},
        {"id": str(uuid.uuid4()), "title": "Normalization"},
        {"id": str(uuid.uuid4()), "title": "Index Optimization"},
    ]

    all_lo = []
    all_lo_kp = []

    for kp in kp_test_data:
        lo_rows, lo_kp_rows = generate_lo_for_kp(kp)
        all_lo.extend(lo_rows)
        all_lo_kp.extend(lo_kp_rows)

    if EXPORT_MODE == "sql":
        export_sql(all_lo, all_lo_kp)
    elif EXPORT_MODE == "csv":
        export_csv(all_lo, all_lo_kp)
    elif EXPORT_MODE == "both":
        export_sql(all_lo, all_lo_kp)
        export_csv(all_lo, all_lo_kp)

if __name__ == "__main__":
    asyncio.run(generate_from_db())
