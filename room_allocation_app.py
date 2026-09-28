import io
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from supabase import Client, create_client


# ==========================================================
# SEMI-AUTOMATIC ROOM ALLOCATION APPLICATION
# ==========================================================
# Preserves the existing Supabase structure:
#   timetable, rooms, labs
#
# Read-only by default. Manual assignments are held in the
# current Streamlit session until the user approves them.
# No Supabase timetable/room writes are performed by this app.
#
# Movement constraints:
#   P1-P2 = one room
#   P3-P4 = one room
#   P5-P7 = one room
# Theory-room changes are permitted only at P1, P3 and P5.
# Laboratory subjects use the fixed room from the labs table.
#
# Room-group policy (nudges which free room is used first):
#   CSC, CSD, IT, EEE1, EEE2       -> B34, B35, B36, A38
#   CAI-1, CAI-2, CSM-1, CSM-2, CSM-3 -> A41, A42, A45, A46, A47
#
# Special rules:
#   - AIT is pinned to B37.
#   - Seminar hall takes up to 4 classes at once, for DTI of
#     ECE-1..4 and CSE-1..4 only.
#   - GROUND is reserved for sports subjects.
#   - PHY and LAC use theory rooms only, never a lab-type room.
#   - B27, B37, C21 (lab-type) may be used only for SINGLE-HOUR
#     theory of CE, AT_I and CP.
#   - Library at P4 and P7 uses B41 / B42 and never a theory room.


st.set_page_config(
    page_title="Semi-Automatic Room Allocation",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
PERIODS = [1, 2, 3, 4, 5, 6, 7]
SHIFT_BLOCKS = {
    1: "P1-P2",
    2: "P1-P2",
    3: "P3-P4",
    4: "P3-P4",
    5: "P5-P7",
    6: "P5-P7",
    7: "P5-P7",
}

TABLE_TIMETABLE = "timetable"
TABLE_ROOMS = "rooms"
TABLE_LABS = "labs"

STATUS_COLORS = {
    "LOCKED": "#7c3aed",
    "APPROVED": "#059669",
    "SUGGESTED": "#2563eb",
    "PENDING": "#d97706",
    "CONFLICT": "#dc2626",
    "LAB FIXED": "#0891b2",
    "UNCHANGED": "#16a34a",
    "ROOM CHANGE": "#d97706",
}

# Class -> preferred room pool. Checked as an exact match first, then as a
# prefix match, so a class code like "CSC-A" still matches the "CSC" rule.
ROOM_GROUP_RULES: List[Tuple[List[str], List[str]]] = [
    (["CSC", "CSD", "IT", "EEE1", "EEE2"], ["B34", "B35", "B36", "A38"]),
    (["CAI-1", "CAI-2", "CSM-1", "CSM-2", "CSM-3"], ["A41", "A42", "A45", "A46", "A47"]),
]

# Ground is reserved for sports subjects only.
GROUND_ROOM = "GROUND"

# Theory rooms that are lab-type. ONLY single-hour theory of the subjects
# in LAB_OVERFLOW_SUBJECTS may use them.
LAB_OVERFLOW_ROOMS_FOR_THEORY = {"B27", "B37", "C21"}
LAB_OVERFLOW_SUBJECTS = {"CE", "A_T", "CP"}

# Never placed in any lab-type room, not even B27/B37/C21.
NO_LAB_SUBJECTS = {"PHY", "LAC"}

# Subjects pinned to one room regardless of the labs table.
FIXED_SUBJECT_ROOMS = {"AIT(B37)": "B37"}

# Seminar hall: shared by several classes at once. The room is found in the
# rooms table by name (any Room_ID containing "SEMINAR").
SEMINAR_KEYWORD = "SEMINAR"
SEMINAR_CAPACITY = 4
SEMINAR_SUBJECTS = {
    "DTI": ["ECE-1", "ECE-2", "ECE-3", "ECE-4", "CSE-1", "CSE-2", "CSE-3", "CSE-4"],
}

# Library at P4 / P7 uses these rooms, never a theory room.
LIBRARY_TOKEN = "LIB"          # subject starts with this (LIB, LIBRARY ...)
LIBRARY_PERIODS = {4, 7}
LIBRARY_ROOMS = ["B41", "B42"]


# ----------------------------------------------------------
# Styling
# ----------------------------------------------------------
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: .1rem;
    }
    .subtitle {
        color: #64748b;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 14px 16px;
        border-radius: 14px;
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15,23,42,.06);
    }
    .room-card {
        padding: 13px;
        margin: 5px 0;
        border-radius: 12px;
        border: 1px solid #dbeafe;
        background: #eff6ff;
    }
    .free-card {
        padding: 13px;
        margin: 5px 0;
        border-radius: 12px;
        border: 1px solid #bbf7d0;
        background: #f0fdf4;
    }
    .hurdle-card {
        padding: 13px;
        margin: 5px 0;
        border-radius: 12px;
        border: 1px solid #fecaca;
        background: #fef2f2;
    }
    .info-card {
        padding: 13px;
        margin: 5px 0;
        border-radius: 12px;
        border: 1px solid #fde68a;
        background: #fffbeb;
    }
    .small-muted {
        color: #64748b;
        font-size: .86rem;
    }
    div[data-testid="stMetric"] {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 8px;
        background: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------
# Utilities / Supabase
# ----------------------------------------------------------
def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    return clean(value).upper()


def env_value(name: str) -> str:
    try:
        value = str(st.secrets.get(name, "")).strip()
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, "").strip()


@st.cache_resource(show_spinner=False)
def get_supabase() -> Client:
    url = env_value("SUPABASE_URL")
    key = env_value("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY environment variables are required."
        )
    return create_client(url, key)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_table(table_name: str) -> pd.DataFrame:
    supabase = get_supabase()
    rows: List[Dict[str, Any]] = []
    start = 0
    page_size = 1000

    while True:
        response = (
            supabase.table(table_name)
            .select("*")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    return pd.DataFrame(rows)


def find_col(df: pd.DataFrame, *names: str) -> str | None:
    if df is None or df.empty:
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        fetch_table(TABLE_TIMETABLE),
        fetch_table(TABLE_ROOMS),
        fetch_table(TABLE_LABS),
    )


# ----------------------------------------------------------
# Normalization — preserves uploaded program's schema logic
# ----------------------------------------------------------
def normalize_timetable(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["Class", "Subject", "Faculty", "Day", "Period", "Room"]
        )

    class_col = find_col(df, "class_id", "class")
    subject_col = find_col(df, "subject_id", "subject")
    faculty_col = find_col(df, "faculty_id", "faculty")
    day_col = find_col(df, "day")
    period_col = find_col(df, "period")
    room_col = find_col(df, "room", "room_id")

    missing = []
    for label, col in [
        ("class_id", class_col),
        ("subject", subject_col),
        ("day", day_col),
        ("period", period_col),
    ]:
        if col is None:
            missing.append(label)

    if missing:
        raise ValueError(
            "timetable table is missing required columns: " + ", ".join(missing)
        )

    out = pd.DataFrame(
        {
            "Class": df[class_col].map(clean),
            "Subject": df[subject_col].map(clean),
            "Faculty": df[faculty_col].map(clean) if faculty_col else "",
            "Day": df[day_col].map(clean).str.title(),
            "Period": pd.to_numeric(df[period_col], errors="coerce"),
            "Room": df[room_col].map(clean) if room_col else "",
        }
    )
    out["Period"] = out["Period"].astype("Int64")
    out = out.dropna(subset=["Period"])
    out = out[out["Day"].isin(DAYS) & out["Period"].isin(PERIODS)]
    out["Period"] = out["Period"].astype(int)
    return out.reset_index(drop=True)


def normalize_rooms(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("rooms table is empty.")

    room_col = find_col(df, "room_id", "room", "room_no", "room_number")
    type_col = find_col(df, "type", "room_type", "category")

    if room_col is None:
        raise ValueError(
            "rooms table must contain Room_ID (or Room / Room_No / Room_Number)."
        )

    out = pd.DataFrame(
        {
            "Room_ID": df[room_col].map(clean),
            "Type": df[type_col].map(clean) if type_col else "",
        }
    )
    out = out[out["Room_ID"] != ""].copy()
    out["_key"] = out["Room_ID"].map(norm)
    out = out.drop_duplicates("_key", keep="first").drop(columns=["_key"])
    return out.reset_index(drop=True)


def normalize_labs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Lab_Subject", "Room"])

    subject_col = find_col(df, "lab_subject", "subject", "lab")
    room_col = find_col(df, "room", "room_id", "room_no", "room_number")

    if subject_col is None or room_col is None:
        return pd.DataFrame(columns=["Lab_Subject", "Room"])

    out = pd.DataFrame(
        {
            "Lab_Subject": df[subject_col].map(clean),
            "Room": df[room_col].map(clean),
        }
    )
    out = out[(out["Lab_Subject"] != "") & (out["Room"] != "")]
    return out.reset_index(drop=True)


def is_lab(subject: str, lab_map: Dict[str, str]) -> bool:
    s = norm(subject)
    return s in lab_map or "LAB" in s


def room_is_lab_type(room_type: str) -> bool:
    t = norm(room_type)
    return any(k in t for k in ["LAB", "LABORATORY", "COMPUTER", "WORKSHOP"])


def room_is_theory_type(room_type: str) -> bool:
    t = norm(room_type)
    return "THEORY" in t or t in {"CLASSROOM", "LECTURE", "LECTURE HALL"}


def preferred_rooms_for_class(cls: str) -> List[str]:
    """Room-group policy lookup. Exact match first, then prefix match, so a
    variant like 'CSC-A' still matches the 'CSC' rule. Returns [] for any
    class not covered."""
    key = norm(cls)
    for class_tokens, room_list in ROOM_GROUP_RULES:
        for token in class_tokens:
            tk = norm(token)
            if key == tk or key.startswith(tk):
                return room_list
    return []


def subject_forbids_labs(subject: str) -> bool:
    """PHY and LAC (and prefixed variants like 'PHY-1') must never be placed
    in any lab-type room, including B27 / B37 / C21."""
    s = norm(subject)
    return any(s == tok or s.startswith(tok) for tok in NO_LAB_SUBJECTS)


def subject_matches(subject: str, tokens) -> bool:
    s = norm(subject)
    return any(s == t or s.startswith(t + "-") or s.startswith(t + " ") for t in tokens)


def subject_may_use_lab_overflow(subject: str) -> bool:
    return subject_matches(subject, LAB_OVERFLOW_SUBJECTS)


def is_seminar_row(subject: str, cls: str) -> bool:
    s, c = norm(subject), norm(cls)
    for code, classes in SEMINAR_SUBJECTS.items():
        if s == code or s.startswith(code + "-"):
            if any(c == norm(x) or c.startswith(norm(x)) for x in classes):
                return True
    return False


def is_library_slot(subject: str, period: int) -> bool:
    return int(period) in LIBRARY_PERIODS and norm(subject).startswith(LIBRARY_TOKEN)


def find_seminar_room(all_rooms) -> str:
    for r in all_rooms:
        if SEMINAR_KEYWORD in norm(r):
            return clean(r)
    return ""


def build_lab_map(labs: pd.DataFrame) -> Dict[str, str]:
    m = {norm(r["Lab_Subject"]): clean(r["Room"]) for _, r in labs.iterrows()}
    for subj, room in FIXED_SUBJECT_ROOMS.items():   # AIT -> B37
        m[norm(subj)] = room
    return m


def is_fixed_row(subject: str, cls: str, lab_map: Dict[str, str]) -> bool:
    return is_lab(subject, lab_map) or is_seminar_row(subject, cls)


def fixed_room_for_row(subject, cls, lab_map, all_rooms) -> str:
    if is_seminar_row(subject, cls):
        return find_seminar_room(all_rooms)
    return lab_map.get(norm(subject), "")


# Faculty names and class-coordinator info both live in the faculty table:
# a faculty_id/name pair, plus a flag for whether that person coordinates a
# class, which class, and their mobile number. Optional — if the table or
# these columns aren't there, the app keeps working and just shows codes.
FACULTY_TABLE_CANDIDATES = ["faculty", "faculty_master", "teachers", "staff"]


@st.cache_data(ttl=30, show_spinner=False)
def fetch_table_optional(table_name: str) -> pd.DataFrame:
    try:
        return fetch_table(table_name)
    except Exception:
        return pd.DataFrame()


def load_faculty_and_coordinators() -> Tuple[Dict[str, str], str, Dict[str, str]]:
    for t in FACULTY_TABLE_CANDIDATES:
        df = fetch_table_optional(t)
        if df.empty:
            continue
        name_col = find_col(df, "faculty_name", "name", "full_name")
        if not name_col:
            continue

        id_col = find_col(df, "faculty_id", "id", "code", "faculty_code", "emp_id", "employee_id")
        faculty_map = (
            {norm(r[id_col]): clean(r[name_col]) for _, r in df.iterrows()} if id_col else {}
        )

        coord_flag_col = find_col(df, "coordinated", "is_coordinator", "coordinator")
        coord_class_col = find_col(
            df, "coordinated_to", "coordinated to", "coordinates", "coordinator_for", "class_coordinated"
        )
        mobile_col = find_col(df, "mobile_number", "mobile number", "mobile", "phone", "phone_number", "contact_number")

        coordinator_map: Dict[str, str] = {}
        if coord_class_col:
            for _, r in df.iterrows():
                cls_val = clean(r[coord_class_col])
                if not cls_val:
                    continue
                if coord_flag_col:
                    flag = norm(r[coord_flag_col])
                    if flag in ("", "NO", "N", "0", "FALSE"):
                        continue
                label = clean(r[name_col])
                if mobile_col:
                    mob = clean(r[mobile_col])
                    if mob:
                        label += f" ({mob})"
                coordinator_map[norm(cls_val)] = label

        if faculty_map or coordinator_map:
            return faculty_map, t, coordinator_map
    return {}, "", {}


def shift_block(period: int) -> str:
    return SHIFT_BLOCKS[int(period)]


def block_periods(block: str) -> List[int]:
    return {
        "P1-P2": [1, 2],
        "P3-P4": [3, 4],
        "P5-P7": [5, 6, 7],
    }.get(block, [])


def slot_key(day: str, period: int) -> Tuple[str, int]:
    return day, int(period)


def assignable_periods(timetable, labs, cls, day, period) -> List[int]:
    """Periods a manual assignment applies to: the movement block's theory
    periods, or just the single period for a library slot."""
    lab_map = build_lab_map(labs)
    sel = timetable[
        (timetable["Class"] == cls)
        & (timetable["Day"] == day)
        & (timetable["Period"] == int(period))
    ]
    if not sel.empty and is_library_slot(sel.iloc[0]["Subject"], period):
        return [int(period)]
    out = []
    for p in block_periods(shift_block(period)):
        r = timetable[
            (timetable["Class"] == cls)
            & (timetable["Day"] == day)
            & (timetable["Period"] == p)
        ]
        if r.empty:
            continue
        s = r.iloc[0]["Subject"]
        if is_fixed_row(s, cls, lab_map) or is_library_slot(s, p):
            continue
        out.append(p)
    return out


# ----------------------------------------------------------
# Session state
# ----------------------------------------------------------
def init_state():
    defaults = {
        "manual_assignments": {},
        "locked_assignments": set(),
        "approved_assignments": set(),
        "rejected_suggestions": set(),
        "last_loaded_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ----------------------------------------------------------
# Automatic suggestions
# ----------------------------------------------------------
def base_room_data(rooms: pd.DataFrame):
    room_records = rooms.to_dict("records")
    all_rooms = [clean(r["Room_ID"]) for r in room_records if clean(r["Room_ID"])]
    room_type = {norm(r["Room_ID"]): clean(r["Type"]) for r in room_records}

    # Never part of the normal theory pool: Ground (sports) and Seminar hall (DTI).
    reserved = {GROUND_ROOM}
    seminar_room = find_seminar_room(all_rooms)
    if seminar_room:
        reserved.add(norm(seminar_room))

    # B27/B37/C21 are kept OUT of the theory pool; they are reachable only
    # through the single-hour CE / AT_I / CP rule.
    theory_rooms = [
        r for r in all_rooms
        if room_is_theory_type(room_type.get(norm(r), ""))
        and norm(r) not in reserved
        and norm(r) not in LAB_OVERFLOW_ROOMS_FOR_THEORY
    ]
    other_rooms = [
        r for r in all_rooms if r not in theory_rooms and norm(r) not in reserved
    ]
    if not theory_rooms:
        theory_rooms = [
            r for r in all_rooms
            if norm(r) not in reserved and norm(r) not in LAB_OVERFLOW_ROOMS_FOR_THEORY
        ]

    lab_overflow_rooms = [
        r for r in other_rooms if norm(r) in LAB_OVERFLOW_ROOMS_FOR_THEORY
    ]
    theory_keys = {norm(x) for x in theory_rooms}
    candidate_base = theory_rooms + [
        r for r in lab_overflow_rooms if norm(r) not in theory_keys
    ]
    return all_rooms, room_type, theory_rooms, candidate_base


def build_fixed_occupancy(
    timetable: pd.DataFrame, rooms: pd.DataFrame, labs: pd.DataFrame
):
    lab_map = build_lab_map(labs)
    all_rooms, room_type, theory_rooms, candidate_base = base_room_data(rooms)
    all_keys = {norm(x) for x in all_rooms}
    occupancy = defaultdict(set)
    shared_count = defaultdict(int)      # seminar hall head-count per slot
    fixed_rows = []
    hurdles = []

    def add_hurdle(row, problem, action):
        hurdles.append(
            {
                "Priority": "HIGH",
                "Day": row["Day"],
                "Period": row["Period"],
                "Class": row["Class"],
                "Subject": row["Subject"],
                "Problem": problem,
                "Action": action,
            }
        )

    for _, row in timetable.sort_values(
        by=["Day", "Period", "Class", "Subject"],
        key=lambda s: s.map(DAYS.index) if s.name == "Day" else s,
    ).iterrows():
        if not is_fixed_row(row["Subject"], row["Class"], lab_map):
            continue

        seminar = is_seminar_row(row["Subject"], row["Class"])
        mapped = fixed_room_for_row(row["Subject"], row["Class"], lab_map, all_rooms)

        if not mapped:
            if seminar:
                add_hurdle(row, "Seminar hall not found in rooms table.",
                           f"Add a room whose ID contains '{SEMINAR_KEYWORD}'.")
            else:
                add_hurdle(row, "Lab subject detected but no lab mapping exists.",
                           "Add the subject-room mapping to the labs table.")
            continue

        if norm(mapped) not in all_keys:
            add_hurdle(row, f"Fixed room '{mapped}' is not in rooms table.",
                       "Check the labs and rooms tables.")
            continue

        key = slot_key(row["Day"], row["Period"])
        if seminar:
            ck = (key, norm(mapped))
            if shared_count[ck] >= SEMINAR_CAPACITY:
                add_hurdle(row,
                           f"Seminar hall capacity ({SEMINAR_CAPACITY} classes) exceeded.",
                           "Move one of the classes to another slot.")
                continue
            shared_count[ck] += 1
        elif norm(mapped) in {norm(x) for x in occupancy[key]}:
            add_hurdle(row, f"Fixed room clash: {mapped}.",
                       "Resolve the clash for this fixed room.")
            continue

        occupancy[key].add(mapped)
        fixed_rows.append(
            {
                "Day": row["Day"],
                "Period": int(row["Period"]),
                "Class": row["Class"],
                "Subject": row["Subject"],
                "Faculty": row["Faculty"],
                "Old Room": row["Room"],
                "Proposed Room": mapped,
                "Allocation": "SEMINAR-FIXED" if seminar else "LAB-FIXED",
                "Shift Block": "SEMINAR" if seminar else "LAB",
                "Status": "LAB FIXED",
                "Reason": "",
            }
        )

    return occupancy, fixed_rows, hurdles, all_rooms, room_type, theory_rooms, candidate_base


def manual_room_for(key):
    return st.session_state.manual_assignments.get(key, "")


def manual_key(day, period, cls):
    return (day, int(period), cls)


def collect_manual_occupancy():
    """
    Returns manual assignments by slot. Only explicitly assigned rooms
    occupy slots. Block consistency is checked separately.
    """
    occ = defaultdict(set)
    for (day, period, cls), room in st.session_state.manual_assignments.items():
        if room:
            occ[slot_key(day, period)].add(room)
    return occ


def suggestion_score(
    room: str,
    cls: str,
    day: str,
    periods: List[int],
    old_common: str,
    previous_room: str,
    same_day_rooms: set,
    proposed: List[Dict[str, Any]],
    theory_rooms: List[str],
    room_type: Dict[str, str],
) -> int:
    rkey = norm(room)
    score = 0
    if old_common and rkey == norm(old_common):
        score += 100000
    if previous_room and rkey == norm(previous_room):
        score += 50000
    if any(rkey == norm(x) for x in same_day_rooms):
        score += 15000
    if room in theory_rooms:
        score += 5000
    usage_hint = sum(
        1
        for x in proposed
        if x["Day"] == day and norm(x["Proposed Room"]) == rkey
    )
    score += min(usage_hint, 20) * 50
    if room_is_lab_type(room_type.get(rkey, "")):
        score -= 2000
    return score


def generate_semi_auto(
    timetable: pd.DataFrame,
    rooms: pd.DataFrame,
    labs: pd.DataFrame,
):
    (
        occupancy,
        proposed,
        hurdles,
        all_rooms,
        room_type,
        theory_rooms,
        candidate_base,
    ) = build_fixed_occupancy(timetable, rooms, labs)

    lab_map = build_lab_map(labs)
    ordered = timetable.sort_values(
        by=["Day", "Period", "Class", "Subject"],
        key=lambda s: s.map(DAYS.index) if s.name == "Day" else s,
    ).copy()

    # Manual assignments have highest priority. They are validated first.
    manual_occ = collect_manual_occupancy()
    for key, rooms_for_slot in manual_occ.items():
        occupancy[key].update(rooms_for_slot)

    # Theory rows: everything that is not fixed (lab / AIT / seminar) and not
    # a library period.
    fixed_or_library = pd.Series(
        [
            is_fixed_row(s, c, lab_map) or is_library_slot(s, p)
            for s, c, p in zip(ordered["Subject"], ordered["Class"], ordered["Period"])
        ],
        index=ordered.index,
    )
    theory_rows = ordered[~fixed_or_library].copy()

    # ---- Library at P4 / P7 -> B41 / B42 only (never a theory room) ----
    room_lookup = {norm(r): r for r in all_rooms}
    lib_pool = [room_lookup[norm(r)] for r in LIBRARY_ROOMS if norm(r) in room_lookup]
    last_library_room: Dict[Tuple[str, str], str] = {}
    is_lib = pd.Series(
        [
            is_library_slot(s, p) and not is_fixed_row(s, c, lab_map)
            for s, c, p in zip(ordered["Subject"], ordered["Class"], ordered["Period"])
        ],
        index=ordered.index,
    )
    for _, x in ordered[is_lib].iterrows():
        day, p, cls = x["Day"], int(x["Period"]), x["Class"]
        old_room = clean(x.get("Room", ""))
        base = {
            "Day": day, "Period": p, "Class": cls, "Subject": x["Subject"],
            "Faculty": x.get("Faculty", ""), "Old Room": old_room,
            "Shift Block": "LIBRARY",
        }
        manual = manual_room_for(manual_key(day, p, cls))
        if manual:
            proposed.append({
                **base, "Proposed Room": manual, "Allocation": "MANUAL",
                "Status": "LOCKED" if (day, p, cls) in st.session_state.locked_assignments
                else "MANUALLY-ASSIGNED",
                "Reason": "",
            })
            continue

        taken = {norm(y) for y in occupancy[slot_key(day, p)]}
        free = [r for r in lib_pool if norm(r) not in taken]
        if not free:
            proposed.append({
                **base, "Proposed Room": "UNALLOCATED", "Allocation": "PENDING",
                "Status": "CONFLICT",
                "Reason": f"{'/'.join(LIBRARY_ROOMS)} are all taken for library at P{p}.",
            })
            hurdles.append({
                "Priority": "HIGH", "Day": day, "Period": p, "Class": cls,
                "Subject": x["Subject"],
                "Problem": f"No library room ({', '.join(LIBRARY_ROOMS)}) free at P{p}.",
                "Action": "Stagger library periods between classes.",
            })
            continue

        prev = last_library_room.get((cls, day), "")
        free.sort(key=lambda r: (
            0 if norm(r) == norm(prev) else 1,
            0 if norm(r) == norm(old_room) else 1,
            lib_pool.index(r),
        ))
        room = free[0]
        occupancy[slot_key(day, p)].add(room)
        last_library_room[(cls, day)] = room
        proposed.append({
            **base, "Proposed Room": room, "Allocation": "AUTO-SUGGESTED",
            "Status": "UNCHANGED" if norm(old_room) == norm(room) else "ROOM CHANGE",
            "Reason": "",
        })

    # Build theory groups, preserving the uploaded program's block constraint.
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for _, row in theory_rows.iterrows():
        groups[(row["Day"], row["Class"], shift_block(int(row["Period"])))].append(
            row.to_dict()
        )

    group_items = sorted(
        groups.items(),
        key=lambda kv: (
            -len(kv[1]),
            DAYS.index(kv[0][0]),
            {"P1-P2": 1, "P3-P4": 2, "P5-P7": 3}.get(kv[0][2], 9),
            kv[0][1],
        ),
    )

    last_room_for_class_day: Dict[Tuple[str, str], str] = {}
    same_class_day_rooms: Dict[Tuple[str, str], set] = defaultdict(set)

    # Seed previous block information from manual assignments.
    for (day, cls, block), group in groups.items():
        block_ps = block_periods(block)
        manual_rooms = []
        for p in block_ps:
            room = manual_room_for(manual_key(day, p, cls))
            if room:
                manual_rooms.append(room)
        if manual_rooms and len({norm(x) for x in manual_rooms}) == 1:
            last_room_for_class_day[(cls, day)] = manual_rooms[0]
            same_class_day_rooms[(cls, day)].add(manual_rooms[0])

    for (day, cls, block), group in group_items:
        periods = sorted({int(x["Period"]) for x in group})
        preferred_rooms = preferred_rooms_for_class(cls)
        old_rooms = [
            clean(x.get("Room", "")) for x in group if clean(x.get("Room", ""))
        ]
        old_common = old_rooms[0] if old_rooms and len({norm(x) for x in old_rooms}) == 1 else ""

        # Check manual assignments in this entire block.
        manual_rooms = [
            manual_room_for(manual_key(day, p, cls)) for p in periods
        ]
        manual_rooms_nonempty = [x for x in manual_rooms if x]
        manual_unique = {norm(x) for x in manual_rooms_nonempty}

        if manual_unique and len(manual_unique) > 1:
            for x in group:
                p = int(x["Period"])
                if not manual_room_for(manual_key(day, p, cls)):
                    room = ""
                else:
                    room = manual_room_for(manual_key(day, p, cls))
                proposed.append(
                    {
                        "Day": day,
                        "Period": p,
                        "Class": cls,
                        "Subject": x["Subject"],
                        "Faculty": x.get("Faculty", ""),
                        "Old Room": x.get("Room", ""),
                        "Proposed Room": room or "CONFLICT",
                        "Allocation": "MANUAL",
                        "Shift Block": block,
                        "Status": "CONFLICT",
                        "Reason": "Different manual rooms were selected within one movement block.",
                    }
                )
                hurdles.append(
                    {
                        "Priority": "HIGH",
                        "Day": day,
                        "Period": p,
                        "Class": cls,
                        "Subject": x["Subject"],
                        "Problem": "Multiple rooms manually selected inside the same movement block.",
                        "Action": f"Use one room for {block}.",
                    }
                )
            continue

        manual_common = manual_rooms_nonempty[0] if manual_rooms_nonempty else ""
        if manual_common:
            # Validate the selected room is free for all periods of the block.
            rkey = norm(manual_common)
            conflicting_periods = [
                p
                for p in periods
                if rkey in {
                    norm(x) for x in occupancy[slot_key(day, p)]
                    if norm(x) != rkey
                }
            ]
            for p in periods:
                fixed_or_manual = {
                    norm(x)
                    for x in occupancy[slot_key(day, p)]
                    if norm(x) != rkey
                }
                if rkey in fixed_or_manual:
                    conflicting_periods.append(p)

            conflicting_periods = sorted(set(conflicting_periods))
            if conflicting_periods:
                for x in group:
                    p = int(x["Period"])
                    proposed.append(
                        {
                            "Day": day,
                            "Period": p,
                            "Class": cls,
                            "Subject": x["Subject"],
                            "Faculty": x.get("Faculty", ""),
                            "Old Room": x.get("Room", ""),
                            "Proposed Room": manual_common,
                            "Allocation": "MANUAL",
                            "Shift Block": block,
                            "Status": "CONFLICT",
                            "Reason": f"Manual room conflicts in period(s): {', '.join('P'+str(p) for p in conflicting_periods)}",
                        }
                    )
                hurdles.append(
                    {
                        "Priority": "HIGH",
                        "Day": day,
                        "Period": periods[0],
                        "Class": cls,
                        "Subject": group[0]["Subject"],
                        "Problem": f"Manual room {manual_common} is occupied during part of {block}.",
                        "Action": "Choose another room for the whole block.",
                    }
                )
                continue

            for x in group:
                p = int(x["Period"])
                occupancy[slot_key(day, p)].add(manual_common)
                proposed.append(
                    {
                        "Day": day,
                        "Period": p,
                        "Class": cls,
                        "Subject": x["Subject"],
                        "Faculty": x.get("Faculty", ""),
                        "Old Room": x.get("Room", ""),
                        "Proposed Room": manual_common,
                        "Allocation": "MANUAL",
                        "Shift Block": block,
                        "Status": "LOCKED" if (day, p, cls) in st.session_state.locked_assignments else "MANUALLY-ASSIGNED",
                        "Reason": "",
                    }
                )
            last_room_for_class_day[(cls, day)] = manual_common
            same_class_day_rooms[(cls, day)].add(manual_common)
            continue

        # Automatic suggestions: find a single room free across the entire block.
        # Room pool for this block:
        #   - PHY / LAC: theory rooms only (never a lab-type room).
        #   - single-hour CE / AT_I / CP: theory rooms + B27/B37/C21.
        #   - everything else: theory rooms only.
        if any(subject_forbids_labs(x["Subject"]) for x in group):
            block_pool = theory_rooms
        elif len(periods) == 1 and all(
            subject_may_use_lab_overflow(x["Subject"]) for x in group
        ):
            block_pool = candidate_base
        else:
            block_pool = theory_rooms

        # Room-group policy: search the class's assigned pool FIRST.
        def score_pool(pool: List[str]) -> List[Tuple[int, str]]:
            found = []
            for idx, room in enumerate(pool):
                rkey = norm(room)
                conflict = any(
                    rkey in {norm(x) for x in occupancy[slot_key(day, p)]}
                    for p in periods
                )
                if conflict:
                    continue
                s = suggestion_score(
                    room,
                    cls,
                    day,
                    periods,
                    old_common,
                    last_room_for_class_day.get((cls, day), ""),
                    same_class_day_rooms[(cls, day)],
                    proposed,
                    theory_rooms,
                    room_type,
                )
                s -= idx
                found.append((s, room))
            return found

        candidates: List[Tuple[int, str]] = []
        if preferred_rooms:
            preferred_set = {norm(p) for p in preferred_rooms}
            group_pool = [r for r in block_pool if norm(r) in preferred_set]
            candidates = score_pool(group_pool)
        if not candidates:
            candidates = score_pool(block_pool)

        if not candidates:
            for x in group:
                p = int(x["Period"])
                proposed.append(
                    {
                        "Day": day,
                        "Period": p,
                        "Class": cls,
                        "Subject": x["Subject"],
                        "Faculty": x.get("Faculty", ""),
                        "Old Room": x.get("Room", ""),
                        "Proposed Room": "UNALLOCATED",
                        "Allocation": "PENDING",
                        "Shift Block": block,
                        "Status": "CONFLICT",
                        "Reason": f"No single room is free for all periods in {block}.",
                    }
                )
            hurdles.append(
                {
                    "Priority": "HIGH",
                    "Day": day,
                    "Period": periods[0],
                    "Class": cls,
                    "Subject": group[0]["Subject"],
                    "Problem": f"No room is free for the whole {block} block.",
                    "Action": "Use the vacant-room finder or move another class.",
                }
            )
            continue

        room = max(candidates, key=lambda z: z[0])[1]
        last_room_for_class_day[(cls, day)] = room
        same_class_day_rooms[(cls, day)].add(room)

        for x in group:
            p = int(x["Period"])
            occupancy[slot_key(day, p)].add(room)
            old_room = clean(x.get("Room", ""))
            status = "UNCHANGED" if old_room and norm(old_room) == norm(room) else "ROOM CHANGE"
            proposed.append(
                {
                    "Day": day,
                    "Period": p,
                    "Class": cls,
                    "Subject": x["Subject"],
                    "Faculty": x.get("Faculty", ""),
                    "Old Room": old_room,
                    "Proposed Room": room,
                    "Allocation": "AUTO-SUGGESTED",
                    "Shift Block": block,
                    "Status": status,
                    "Reason": "",
                }
            )

    result = pd.DataFrame(proposed)
    if result.empty:
        result = pd.DataFrame(
            columns=[
                "Day", "Period", "Class", "Subject", "Faculty", "Old Room",
                "Proposed Room", "Allocation", "Shift Block", "Status", "Reason",
            ]
        )

    day_order = {d: i for i, d in enumerate(DAYS)}
    result["_day_order"] = result["Day"].map(day_order)
    result = (
        result.sort_values(["_day_order", "Class", "Period", "Subject"])
        .drop(columns=["_day_order"])
        .reset_index(drop=True)
    )

    return result, hurdles


# ----------------------------------------------------------
# Room suggestions / vacancies / validation
# ----------------------------------------------------------
def fixed_room_occupancy(timetable, rooms, labs):
    occupancy, fixed_rows, hurdles, *_ = build_fixed_occupancy(timetable, rooms, labs)
    return occupancy, fixed_rows, hurdles


def vacant_rooms_for(
    day: str,
    period: int,
    timetable: pd.DataFrame,
    rooms: pd.DataFrame,
    labs: pd.DataFrame,
    exclude_class: str | None = None,
):
    fixed_occ, _, fixed_hurdles, all_rooms, room_type, theory_rooms, _ = build_fixed_occupancy(
        timetable, rooms, labs
    )

    occupied = {norm(x) for x in fixed_occ[slot_key(day, period)]}

    # Include current manual assignments, but ignore the selected class itself.
    for (d, p, cls), room in st.session_state.manual_assignments.items():
        if d == day and int(p) == int(period) and cls != exclude_class and room:
            occupied.add(norm(room))

    available = [r for r in all_rooms if norm(r) not in occupied]
    return available, occupied, room_type, fixed_hurdles


def class_block_for_selection(timetable, cls, day, period):
    rows = timetable[
        (timetable["Class"] == cls)
        & (timetable["Day"] == day)
        & (timetable["Period"] == int(period))
    ]
    if rows.empty:
        return shift_block(period), "", False
    row = rows.iloc[0]
    return shift_block(period), row["Subject"], is_lab(row["Subject"], build_lab_map(st.session_state.labs))


def validate_manual_assignment(
    cls: str,
    day: str,
    period: int,
    room: str,
    timetable: pd.DataFrame,
    rooms: pd.DataFrame,
    labs: pd.DataFrame,
):
    room = clean(room)
    if not room:
        return False, "No room selected."

    room_ids = list(rooms["Room_ID"])
    if norm(room) not in {norm(x) for x in room_ids}:
        return False, f"{room} is not present in the rooms table."

    rows = timetable[
        (timetable["Class"] == cls)
        & (timetable["Day"] == day)
        & (timetable["Period"] == int(period))
    ]
    if rows.empty:
        return False, "No timetable row exists for this class/day/period."

    subject = rows.iloc[0]["Subject"]
    lab_map = build_lab_map(labs)

    # Labs, AIT and seminar subjects are fixed by rule.
    if is_fixed_row(subject, cls, lab_map):
        fixed = fixed_room_for_row(subject, cls, lab_map, room_ids)
        return False, (
            f"{subject} is a fixed subject placed in {fixed or 'its fixed room'} "
            "automatically; it cannot be reassigned manually."
        )

    seminar_room = find_seminar_room(room_ids)
    library = is_library_slot(subject, period)

    if library:
        if norm(room) not in {norm(x) for x in LIBRARY_ROOMS}:
            return False, f"Library periods (P4/P7) must use {', '.join(LIBRARY_ROOMS)}."
    else:
        if seminar_room and norm(room) == norm(seminar_room):
            return False, "Seminar hall is reserved for DTI (ECE/CSE 1-4)."
        if norm(room) == GROUND_ROOM:
            return False, "Ground is reserved for sports subjects and cannot be used for theory."

    room_type_map = {norm(r): t for r, t in zip(rooms["Room_ID"], rooms["Type"])}
    is_lab_room = (
        room_is_lab_type(room_type_map.get(norm(room), ""))
        or norm(room) in LAB_OVERFLOW_ROOMS_FOR_THEORY
    )
    if is_lab_room and not library:
        if subject_forbids_labs(subject):
            return False, f"{subject} must not be placed in a lab room, even {room}."
        if norm(room) not in LAB_OVERFLOW_ROOMS_FOR_THEORY:
            return False, (
                f"{room} is a lab-type room. Only "
                f"{', '.join(sorted(LAB_OVERFLOW_ROOMS_FOR_THEORY))} may be used, and only "
                f"for single-hour {', '.join(sorted(LAB_OVERFLOW_SUBJECTS))}."
            )
        if not subject_may_use_lab_overflow(subject):
            return False, (
                f"Only single-hour {', '.join(sorted(LAB_OVERFLOW_SUBJECTS))} theory "
                f"may use {room}."
            )

    periods = assignable_periods(timetable, labs, cls, day, period)
    block = "LIBRARY" if library else shift_block(period)

    if is_lab_room and not library and len(periods) > 1:
        return False, f"{room} is only for single-hour theory; this block has {len(periods)} periods."

    for p in periods:
        available_p, _, _, _ = vacant_rooms_for(
            day, p, timetable, rooms, labs, exclude_class=cls
        )
        if norm(room) not in {norm(x) for x in available_p}:
            return False, f"{room} is occupied/reserved at {day} P{p}."

    return True, f"Valid for {block}. The class can remain in {room} for this period/block."


def render_class_grid(proposed_df: pd.DataFrame, cls: str) -> str:
    """Builds a plain-language day-by-period timetable grid (HTML) for one
    class, colour-coded by the same STATUS_COLORS used everywhere else."""
    sub = proposed_df[proposed_df["Class"] == cls]
    header_cells = "".join(
        f"<th style='padding:7px;border:1px solid #e2e8f0;background:#f8fafc;font-size:.8rem'>P{p}</th>"
        for p in PERIODS
    )
    header = (
        "<tr><th style='padding:7px;border:1px solid #e2e8f0;background:#f8fafc;"
        f"font-size:.8rem;text-align:left'>Day</th>{header_cells}</tr>"
    )
    body_rows = []
    for day in DAYS:
        cells = []
        for p in PERIODS:
            hit = sub[(sub["Day"] == day) & (sub["Period"] == p)]
            if hit.empty:
                cells.append(
                    "<td style='padding:7px;border:1px solid #e2e8f0;color:#cbd5e1;"
                    "text-align:center'>—</td>"
                )
                continue
            z = hit.iloc[0]
            color = STATUS_COLORS.get(z["Status"], "#64748b")
            cells.append(
                "<td style='padding:7px;border:1px solid #e2e8f0;text-align:center;font-size:.78rem'>"
                f"<b>{z['Subject']}</b><br>"
                f"<span style='color:#64748b'>{z['Faculty'] or '—'}</span><br>"
                "<span style='display:inline-block;margin-top:3px;padding:1px 8px;"
                f"border-radius:10px;background:{color}22;color:{color};font-weight:700'>"
                f"{z['Proposed Room']}</span></td>"
            )
        body_rows.append(
            "<tr><td style='padding:7px;border:1px solid #e2e8f0;font-weight:700;"
            f"background:#f8fafc'>{day}</td>{''.join(cells)}</tr>"
        )
    return (
        "<table style='border-collapse:collapse;width:100%'>"
        f"{header}{''.join(body_rows)}</table>"
    )


def class_day_mapping(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return pd.DataFrame()

    rows = []
    for (cls, day), g in result.groupby(["Class", "Day"], sort=False):
        row = {"Day": day, "Class": cls}
        for p in PERIODS:
            gp = g[g["Period"] == p]
            if gp.empty:
                row[f"P{p}"] = "—"
            else:
                x = gp.iloc[0]
                room = x["Proposed Room"]
                if x["Allocation"] == "LAB-FIXED":
                    row[f"P{p}"] = f"{room} [LAB]"
                elif x["Allocation"] == "SEMINAR-FIXED":
                    row[f"P{p}"] = f"{room} [SEM]"
                elif room in ("UNALLOCATED", "CONFLICT"):
                    row[f"P{p}"] = room
                else:
                    row[f"P{p}"] = room
        rows.append(row)

    out = pd.DataFrame(rows)
    out["_day"] = out["Day"].map({d: i for i, d in enumerate(DAYS)})
    return out.sort_values(["_day", "Class"]).drop(columns=["_day"]).reset_index(drop=True)


def room_occupancy_grid(result: pd.DataFrame, rooms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for room in rooms["Room_ID"].tolist():
        row = {"Room": room}
        for d in DAYS:
            for p in PERIODS:
                x = result[
                    (result["Day"] == d)
                    & (result["Period"] == p)
                    & (result["Proposed Room"].map(norm) == norm(room))
                ]
                if x.empty:
                    row[f"{d[:3]} P{p}"] = "🟢 FREE"
                elif len(x) > 1:
                    row[f"{d[:3]} P{p}"] = " + ".join(
                        f"{z['Class']}/{z['Subject']}" for _, z in x.iterrows()
                    )
                else:
                    z = x.iloc[0]
                    row[f"{d[:3]} P{p}"] = f"{z['Class']} / {z['Subject']}"
        rows.append(row)
    return pd.DataFrame(rows)


def vacant_grid(
    result: pd.DataFrame, rooms: pd.DataFrame, day: str
) -> pd.DataFrame:
    rows = []
    for p in PERIODS:
        occupied = set(
            result[
                (result["Day"] == day) & (result["Period"] == p)
            ]["Proposed Room"].map(norm).tolist()
        )
        for room in rooms["Room_ID"].tolist():
            if norm(room) not in occupied:
                rows.append(
                    {"Period": f"P{p}", "Room": room, "Status": "🟢 VACANT"}
                )
    return pd.DataFrame(rows)


def to_excel(
    timetable: pd.DataFrame,
    proposed: pd.DataFrame,
    mapping: pd.DataFrame,
    rooms: pd.DataFrame,
    hurdles: pd.DataFrame,
    manual: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        timetable.to_excel(writer, sheet_name="Current_Timetable", index=False)
        proposed.to_excel(writer, sheet_name="Proposed_Allocation", index=False)
        mapping.to_excel(writer, sheet_name="Class_Room_Mapping", index=False)
        rooms.to_excel(writer, sheet_name="Rooms_Master", index=False)
        hurdles.to_excel(writer, sheet_name="Hurdles", index=False)
        manual.to_excel(writer, sheet_name="Manual_Assignments", index=False)
    output.seek(0)
    return output.getvalue()


# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
st.markdown('<div class="main-title">🏫 Semi-Automatic Room Allocation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Suggest → check constraints → manually adjust → lock/approve → export</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔄 Reload Supabase data", use_container_width=True):
        fetch_table.clear()
        st.session_state.manual_assignments = {}
        st.session_state.locked_assignments = set()
        st.session_state.approved_assignments = set()
        st.rerun()

    st.markdown("### Movement constraints")
    st.info(
        "P1–P2: one room\n\n"
        "P3–P4: one room\n\n"
        "P5–P7: one room\n\n"
        "Room changes are permitted only at P1, P3 and P5."
    )

    st.markdown("### Allocation logic")
    st.write("🔵 Labs use the fixed room in `labs`.")
    st.write("🟢 Theory subjects use theory rooms.")
    st.write("🔴 Conflicts are highlighted.")
    st.write("🟣 Locked manual assignments are preserved.")

    st.markdown("### Room-group policy")
    st.write("🏢 CSC · CSD · IT · EEE1 · EEE2 → B34, B35, B36, A38 (when free)")
    st.write("🏢 CAI-1 · CAI-2 · CSM-1 · CSM-2 · CSM-3 → A41, A42, A45, A46, A47 (when free)")
    st.write("↩️ Every other class, and any overflow, uses the next best available theory room.")

    st.markdown("### Special room rules")
    st.write("🏟️ Ground is reserved for sports subjects — never offered to theory.")
    st.write("🚫 PHY and LAC use theory rooms only, never any lab room.")
    st.write("🧪 B27/B37/C21: only single-hour CE, AT_I, CP theory.")
    st.write("🏛️ Seminar hall: DTI for ECE-1..4 / CSE-1..4 (max 4 classes).")
    st.write("📚 Library at P4/P7 → B41/B42 only.")
    st.write("📌 AIT → B37 always.")

try:
    with st.spinner("Reading timetable, rooms and lab information from Supabase..."):
        raw_timetable, raw_rooms, raw_labs = load_data()
        timetable = normalize_timetable(raw_timetable)
        rooms = normalize_rooms(raw_rooms)
        labs = normalize_labs(raw_labs)
        st.session_state.labs = labs
except Exception as exc:
    st.error(f"Could not read Supabase data: {exc}")
    st.info("Check SUPABASE_URL, SUPABASE_KEY, and the table/column names.")
    st.stop()

if timetable.empty:
    st.warning("No timetable rows were found in the Supabase 'timetable' table.")
    st.stop()

# Optional lookups — never fatal if the tables don't exist.
faculty_map, faculty_table_used, coordinator_map = load_faculty_and_coordinators()

if faculty_map:
    timetable["Faculty"] = timetable["Faculty"].map(
        lambda code: faculty_map.get(norm(code), code) if code else code
    )

with st.sidebar:
    st.markdown("### Faculty & coordinator lookup")
    if faculty_map:
        st.write(f"✅ Faculty names loaded from `{faculty_table_used}` ({len(faculty_map)} entries).")
    else:
        st.write(
            "⚠️ No faculty-name table found (tried: faculty, faculty_master, teachers, staff). "
            "Faculty codes are shown as-is — tell me your table/column names and I'll wire it up."
        )
    if coordinator_map:
        st.write(f"✅ Class coordinators loaded from `{faculty_table_used}` ({len(coordinator_map)} entries).")
    else:
        st.write(
            "⚠️ No coordinator info found in the faculty table (tried columns: coordinated, "
            "coordinated_to, mobile_number and close variants). Tell me the exact column names "
            "and I'll wire it up."
        )

# ----------------------------------------------------------
# Generate current suggestions
# ----------------------------------------------------------
try:
    proposed, hurdle_rows = generate_semi_auto(timetable, rooms, labs)
except Exception as exc:
    st.error(f"Room analysis failed: {exc}")
    st.exception(exc)
    st.stop()

hurdles = pd.DataFrame(
    hurdle_rows,
    columns=["Priority", "Day", "Period", "Class", "Subject", "Problem", "Action"],
)

allocated = int(
    proposed["Proposed Room"].notna().sum()
    - (proposed["Proposed Room"].isin(["UNALLOCATED", "CONFLICT"])).sum()
)
pending = int(
    proposed["Proposed Room"].isin(["UNALLOCATED", "CONFLICT"]).sum()
)
locked = len(st.session_state.locked_assignments)
approved = len(st.session_state.approved_assignments)
room_changes = int((proposed["Status"] == "ROOM CHANGE").sum())
rooms_used = (
    proposed[~proposed["Proposed Room"].isin(["UNALLOCATED", "CONFLICT"])]
    ["Proposed Room"].nunique()
    if not proposed.empty else 0
)

# ----------------------------------------------------------
# Dashboard metrics
# ----------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("📚 Timetable periods", len(timetable))
c2.metric("🟢 Suggested/assigned", allocated)
c3.metric("🔴 Pending", pending)
c4.metric("⚠️ Hurdles", len(hurdles))
c5.metric("🔒 Locked", locked)
c6.metric("🏫 Rooms used", rooms_used)

if pending == 0 and hurdles.empty:
    st.success("All timetable periods have a room suggestion/assignment and no unresolved hurdles were detected.")
elif pending:
    st.warning(f"{pending} timetable periods need manual attention.")
else:
    st.info("Room suggestions are available. Review the hurdle list before approval.")

# ----------------------------------------------------------
# Main navigation
# ----------------------------------------------------------
tabs = st.tabs(
    [
        "📊 Dashboard",
        "📅 Class View",
        "🗓️ Timetable View",
        "🛠️ Manual Assignment",
        "🟢 Vacant Rooms",
        "⚠️ Hurdles",
        "🏫 Room View",
        "📋 Allocation Table",
        "📥 Export",
    ]
)

# ----------------------------------------------------------
# Dashboard
# ----------------------------------------------------------
with tabs[0]:
    st.subheader("Allocation overview")

    a, b, c = st.columns(3)
    with a:
        st.markdown("#### Status")
        status_counts = proposed["Status"].value_counts().rename_axis("Status").reset_index(name="Count")
        st.dataframe(status_counts, hide_index=True, use_container_width=True)

    with b:
        st.markdown("#### Allocation type")
        type_counts = proposed["Allocation"].value_counts().rename_axis("Allocation").reset_index(name="Count")
        st.dataframe(type_counts, hide_index=True, use_container_width=True)

    with c:
        st.markdown("#### Room usage")
        room_counts = (
            proposed[
                ~proposed["Proposed Room"].isin(["UNALLOCATED", "CONFLICT"])
            ]["Proposed Room"]
            .value_counts()
            .rename_axis("Room")
            .reset_index(name="Periods")
        )
        st.dataframe(room_counts.head(12), hide_index=True, use_container_width=True)

    st.markdown("### 🚦 What needs attention?")
    if hurdles.empty:
        st.success("No unresolved automatic hurdles.")
    else:
        st.dataframe(hurdles, use_container_width=True, hide_index=True)

    st.markdown("### 🧭 Rule reminder")
    st.info(
        "A class gets one room across P1–P2, one across P3–P4 and one across P5–P7. "
        "Named class groups are nudged toward their assigned room pool. AIT is always in B37. "
        "DTI for ECE/CSE 1–4 uses the seminar hall (max 4 classes at a time). Library at P4/P7 "
        "uses B41/B42 only. PHY and LAC use theory rooms only. B27/B37/C21 are used only for "
        "single-hour CE, AT_I and CP theory. Ground is for sports only."
    )

# ----------------------------------------------------------
# Class view
# ----------------------------------------------------------
with tabs[1]:
    st.subheader("📅 Class-wise room schedule")

    class_options = ["All"] + sorted(timetable["Class"].unique().tolist())
    day_options = ["All"] + DAYS

    f1, f2 = st.columns(2)
    selected_class = f1.selectbox("Class", class_options, key="class_view_class")
    selected_day = f2.selectbox("Day", day_options, key="class_view_day")

    display = proposed.copy()
    if selected_class != "All":
        display = display[display["Class"] == selected_class]
    if selected_day != "All":
        display = display[display["Day"] == selected_day]

    mapping = class_day_mapping(display)
    if mapping.empty:
        st.info("No entries for the selected filter.")
    else:
        st.dataframe(mapping, use_container_width=True, hide_index=True)

    st.markdown("### Detailed view")
    cols = [
        "Day", "Period", "Class", "Subject", "Faculty",
        "Old Room", "Proposed Room", "Allocation",
        "Shift Block", "Status", "Reason",
    ]
    st.dataframe(display[cols], use_container_width=True, hide_index=True, height=430)

# ----------------------------------------------------------
# Timetable view (grid) with download
# ----------------------------------------------------------
with tabs[2]:
    st.subheader("🗓️ Timetable view")
    st.caption(
        "A day-by-period grid for one class — the same shape as a printed timetable — "
        "built from the current suggestions and any manual overrides."
    )

    tv_class = st.selectbox(
        "Class", sorted(timetable["Class"].unique().tolist()), key="tv_class"
    )
    st.markdown(f"**Class coordinator:** {coordinator_map.get(norm(tv_class), 'Not available')}")

    st.markdown(render_class_grid(proposed, tv_class), unsafe_allow_html=True)

    st.markdown("#### Legend")
    legend_cols = st.columns(len(STATUS_COLORS))
    for col, (status, color) in zip(legend_cols, STATUS_COLORS.items()):
        col.markdown(
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{color};margin-right:6px'></span>{status}",
            unsafe_allow_html=True,
        )

    tv_data = proposed[proposed["Class"] == tv_class][
        [
            "Day", "Period", "Subject", "Faculty",
            "Old Room", "Proposed Room", "Allocation", "Status", "Reason",
        ]
    ].reset_index(drop=True)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        f"⬇️ Download {tv_class} timetable (CSV)",
        tv_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{tv_class}_timetable.csv",
        mime="text/csv",
        use_container_width=True,
        key="tv_dl_csv",
    )
    tv_excel = io.BytesIO()
    with pd.ExcelWriter(tv_excel, engine="openpyxl") as writer:
        tv_data.to_excel(writer, sheet_name=(tv_class[:31] or "Timetable"), index=False)
    dl2.download_button(
        f"📊 Download {tv_class} timetable (Excel)",
        tv_excel.getvalue(),
        file_name=f"{tv_class}_timetable.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="tv_dl_xlsx",
    )

# ----------------------------------------------------------
# Manual assignment
# ----------------------------------------------------------
with tabs[3]:
    st.subheader("🛠️ Manual room assignment")
    st.caption(
        "Choose a class, day and period. The application suggests vacant rooms and checks "
        "the complete movement block before allowing the assignment. All room rules "
        "(Ground, seminar hall, lab rooms, library, PHY/LAC) are enforced here too."
    )

    m1, m2, m3 = st.columns(3)
    manual_class = m1.selectbox(
        "Class", sorted(timetable["Class"].unique().tolist()), key="manual_class"
    )
    manual_day = m2.selectbox("Day", DAYS, key="manual_day")
    manual_period = m3.selectbox("Period", PERIODS, key="manual_period")

    row = timetable[
        (timetable["Class"] == manual_class)
        & (timetable["Day"] == manual_day)
        & (timetable["Period"] == manual_period)
    ]

    if row.empty:
        st.warning("No timetable entry exists for this class/day/period.")
    else:
        r = row.iloc[0]
        block = shift_block(manual_period)
        lab_map_now = build_lab_map(labs)
        library_now = is_library_slot(r["Subject"], manual_period)
        fixed_now = is_fixed_row(r["Subject"], manual_class, lab_map_now)
        target_periods = assignable_periods(
            timetable, labs, manual_class, manual_day, manual_period
        )
        st.markdown(
            f"**Subject:** {r['Subject']} &nbsp;&nbsp; | &nbsp;&nbsp; "
            f"**Faculty:** {r['Faculty']} &nbsp;&nbsp; | &nbsp;&nbsp; "
            f"**Movement block:** `{'LIBRARY' if library_now else block}`"
        )

        available, occupied, room_type, fixed_hurdles = vacant_rooms_for(
            manual_day,
            manual_period,
            timetable,
            rooms,
            labs,
            exclude_class=manual_class,
        )

        # Existing selection
        existing = manual_room_for(
            manual_key(manual_day, manual_period, manual_class)
        )

        # Room-group policy: list this class's preferred pool first.
        preferred_for_class = preferred_rooms_for_class(manual_class)
        if library_now:
            preferred_for_class = LIBRARY_ROOMS
        if preferred_for_class:
            preferred_set = {norm(p) for p in preferred_for_class}
            available_ordered = [r2 for r2 in available if norm(r2) in preferred_set] + [
                r2 for r2 in available if norm(r2) not in preferred_set
            ]
        else:
            available_ordered = available

        room_options = ["-- Select room --"] + available_ordered
        if existing and existing not in room_options:
            room_options.insert(1, existing)

        selected_room = st.selectbox(
            "Available / suggested rooms",
            room_options,
            index=room_options.index(existing) if existing in room_options else 0,
            key="manual_room_selector",
        )

        if preferred_for_class:
            st.caption(
                f"Preferred rooms for {manual_class}: {', '.join(preferred_for_class)} "
                "(listed first above, when free)."
            )

        if fixed_now:
            st.caption(
                "📌 This is a fixed subject (lab / AIT / seminar). Its room is set "
                "automatically and cannot be changed manually."
            )
        elif library_now:
            st.caption(f"📚 Library period — only {', '.join(LIBRARY_ROOMS)} are allowed.")
        elif subject_forbids_labs(r["Subject"]):
            st.caption(f"⚠️ {r['Subject']} must not be assigned to any lab room, including B27/B37/C21.")
        else:
            st.caption(
                "ℹ️ Theory subject — Ground and the seminar hall are unavailable. "
                f"{', '.join(sorted(LAB_OVERFLOW_ROOMS_FOR_THEORY))} are only for single-hour "
                f"{', '.join(sorted(LAB_OVERFLOW_SUBJECTS))}."
            )

        if available:
            st.success(
                f"🟢 {len(available)} rooms are vacant at {manual_day} P{manual_period}."
            )
        else:
            st.error("🔴 No vacant rooms are available for this period.")

        # Suggested room from automatic result
        auto_row = proposed[
            (proposed["Class"] == manual_class)
            & (proposed["Day"] == manual_day)
            & (proposed["Period"] == manual_period)
        ]
        if not auto_row.empty:
            suggested = auto_row.iloc[0]["Proposed Room"]
            if suggested not in ("UNALLOCATED", "CONFLICT"):
                st.info(f"💡 Automatic suggestion: **{suggested}**")

        v1, v2 = st.columns(2)
        with v1:
            if selected_room != "-- Select room --":
                valid, message = validate_manual_assignment(
                    manual_class,
                    manual_day,
                    manual_period,
                    selected_room,
                    timetable,
                    rooms,
                    labs,
                )
                if valid:
                    st.success("✅ " + message)
                else:
                    st.error("❌ " + message)

        with v2:
            st.markdown("**Periods affected:** " + (", ".join(f"P{p}" for p in target_periods) or "—"))

        b1, b2, b3 = st.columns(3)
        if b1.button("💾 Assign / Update", use_container_width=True, type="primary"):
            if selected_room == "-- Select room --":
                st.error("Select a room first.")
            else:
                valid, message = validate_manual_assignment(
                    manual_class,
                    manual_day,
                    manual_period,
                    selected_room,
                    timetable,
                    rooms,
                    labs,
                )
                if not valid:
                    st.error(message)
                else:
                    for p in assignable_periods(timetable, labs, manual_class, manual_day, manual_period):
                        st.session_state.manual_assignments[
                            manual_key(manual_day, p, manual_class)
                        ] = selected_room
                    st.success(
                        f"Assigned {selected_room} to {manual_class} for {manual_day} "
                        f"{'P' + str(manual_period) if library_now else block}."
                    )
                    st.rerun()

        if b2.button("🔒 Lock block", use_container_width=True):
            room = selected_room
            if room == "-- Select room --":
                st.error("Select a room first.")
            else:
                valid, message = validate_manual_assignment(
                    manual_class,
                    manual_day,
                    manual_period,
                    room,
                    timetable,
                    rooms,
                    labs,
                )
                if not valid:
                    st.error(message)
                else:
                    for p in assignable_periods(timetable, labs, manual_class, manual_day, manual_period):
                        k = manual_key(manual_day, p, manual_class)
                        st.session_state.manual_assignments[k] = room
                        st.session_state.locked_assignments.add(k)
                    st.success(
                        f"🔒 {manual_class} is locked in {room} for "
                        f"{'P' + str(manual_period) if library_now else block}."
                    )
                    st.rerun()

        if b3.button("↩️ Clear block", use_container_width=True):
            for p in assignable_periods(timetable, labs, manual_class, manual_day, manual_period):
                k = manual_key(manual_day, p, manual_class)
                st.session_state.manual_assignments.pop(k, None)
                st.session_state.locked_assignments.discard(k)
                st.session_state.approved_assignments.discard(k)
            st.success(f"Cleared manual assignment for {manual_class}, {manual_day}.")
            st.rerun()

    st.markdown("### Current manual assignments")
    if st.session_state.manual_assignments:
        manual_rows = []
        for (d, p, cls), room in sorted(
            st.session_state.manual_assignments.items(),
            key=lambda x: (DAYS.index(x[0][0]), x[0][2], x[0][1]),
        ):
            manual_rows.append(
                {
                    "Day": d,
                    "Period": p,
                    "Class": cls,
                    "Room": room,
                    "Locked": (d, p, cls) in st.session_state.locked_assignments,
                }
            )
        st.dataframe(pd.DataFrame(manual_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No manual assignments yet.")

# ----------------------------------------------------------
# Vacant rooms
# ----------------------------------------------------------
with tabs[4]:
    st.subheader("🟢 Vacant classroom finder")

    v1, v2 = st.columns(2)
    vacancy_day = v1.selectbox("Day", DAYS, key="vacancy_day")
    vacancy_period = v2.selectbox("Period", PERIODS, key="vacancy_period")

    available, occupied, room_type, _ = vacant_rooms_for(
        vacancy_day, vacancy_period, timetable, rooms, labs
    )

    st.markdown(
        f"### {len(available)} vacant rooms at {vacancy_day} – P{vacancy_period}"
    )

    if available:
        card_cols = st.columns(4)
        for i, room in enumerate(available):
            with card_cols[i % 4]:
                st.markdown(
                    f'<div class="free-card"><b>🟢 {room}</b><br>'
                    f'<span class="small-muted">{room_type.get(norm(room), "Room")}</span></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.error("No vacant rooms for this slot.")

    st.markdown("### Occupied rooms")
    occupied_rows = []
    for room in rooms["Room_ID"]:
        if norm(room) in occupied:
            hit = proposed[
                (proposed["Day"] == vacancy_day)
                & (proposed["Period"] == vacancy_period)
                & (proposed["Proposed Room"].map(norm) == norm(room))
            ]
            if hit.empty:
                who = "Fixed/reserved"
            else:
                who = ", ".join(f"{z['Class']} – {z['Subject']}" for _, z in hit.iterrows())
            occupied_rows.append(
                {
                    "Room": room,
                    "Type": room_type.get(norm(room), ""),
                    "Status": "🔴 OCCUPIED",
                    "Class / Purpose": who,
                }
            )
    st.dataframe(
        pd.DataFrame(occupied_rows),
        use_container_width=True,
        hide_index=True,
    )

# ----------------------------------------------------------
# Hurdles
# ----------------------------------------------------------
with tabs[5]:
    st.subheader("⚠️ Hurdle identification")

    if hurdles.empty:
        st.success("No unresolved hurdles were detected.")
    else:
        high = int((hurdles["Priority"] == "HIGH").sum())
        st.warning(f"{len(hurdles)} hurdles detected ({high} high priority).")

        st.dataframe(
            hurdles,
            use_container_width=True,
            hide_index=True,
            height=520,
        )

        st.markdown("### Hurdle categories")
        categories = []
        for _, h in hurdles.iterrows():
            problem = str(h["Problem"])
            if "No room" in problem or "No library room" in problem:
                cat = "Room shortage"
            elif "capacity" in problem.lower():
                cat = "Seminar hall capacity"
            elif "conflict" in problem.lower() or "clash" in problem.lower():
                cat = "Room conflict"
            elif "lab" in problem.lower():
                cat = "Laboratory constraint"
            elif "movement" in problem.lower() or "block" in problem.lower():
                cat = "Movement constraint"
            else:
                cat = "Other"
            categories.append(cat)
        hc = pd.Series(categories).value_counts().rename_axis("Category").reset_index(name="Count")
        st.dataframe(hc, use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# Room view
# ----------------------------------------------------------
with tabs[6]:
    st.subheader("🏫 Room-wise occupancy")

    rv1, rv2 = st.columns(2)
    selected_room = rv1.selectbox(
        "Room", rooms["Room_ID"].tolist(), key="room_view_room"
    )
    selected_room_day = rv2.selectbox("Day", DAYS, key="room_view_day")

    room_rows = []
    for p in PERIODS:
        hit = proposed[
            (proposed["Day"] == selected_room_day)
            & (proposed["Period"] == p)
            & (proposed["Proposed Room"].map(norm) == norm(selected_room))
        ]
        if hit.empty:
            room_rows.append(
                {"Period": f"P{p}", "Status": "🟢 FREE", "Class": "", "Subject": ""}
            )
        else:
            room_rows.append(
                {
                    "Period": f"P{p}",
                    "Status": "🔴 OCCUPIED",
                    "Class": ", ".join(hit["Class"].astype(str)),
                    "Subject": ", ".join(hit["Subject"].astype(str)),
                }
            )
    st.dataframe(pd.DataFrame(room_rows), use_container_width=True, hide_index=True)

    st.markdown("### Complete room occupancy matrix")
    st.dataframe(
        room_occupancy_grid(proposed, rooms),
        use_container_width=True,
        hide_index=True,
        height=520,
    )

# ----------------------------------------------------------
# Allocation table
# ----------------------------------------------------------
with tabs[7]:
    st.subheader("📋 Allocation table")

    filter1, filter2, filter3 = st.columns(3)
    f_class = filter1.selectbox(
        "Class", ["All"] + sorted(proposed["Class"].unique().tolist()), key="allocation_class"
    )
    f_day = filter2.selectbox("Day", ["All"] + DAYS, key="allocation_day")
    f_status = filter3.selectbox(
        "Status",
        ["All"] + sorted(proposed["Status"].dropna().unique().tolist()),
        key="allocation_status",
    )

    alloc = proposed.copy()
    if f_class != "All":
        alloc = alloc[alloc["Class"] == f_class]
    if f_day != "All":
        alloc = alloc[alloc["Day"] == f_day]
    if f_status != "All":
        alloc = alloc[alloc["Status"] == f_status]

    st.dataframe(
        alloc[
            [
                "Day", "Period", "Class", "Subject", "Faculty",
                "Old Room", "Proposed Room", "Allocation",
                "Shift Block", "Status", "Reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=620,
    )

# ----------------------------------------------------------
# Export
# ----------------------------------------------------------
with tabs[8]:
    st.subheader("📥 Export / review")

    manual_rows = []
    for (d, p, cls), room in st.session_state.manual_assignments.items():
        manual_rows.append(
            {
                "Day": d,
                "Period": p,
                "Class": cls,
                "Room": room,
                "Locked": (d, p, cls) in st.session_state.locked_assignments,
                "Approved": (d, p, cls) in st.session_state.approved_assignments,
            }
        )
    manual_df = pd.DataFrame(
        manual_rows,
        columns=["Day", "Period", "Class", "Room", "Locked", "Approved"],
    )

    mapping = class_day_mapping(proposed)

    st.markdown("### Approval summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Locked assignments", len(st.session_state.locked_assignments))
    s2.metric("Approved assignments", len(st.session_state.approved_assignments))
    s3.metric("Remaining hurdles", len(hurdles))

    st.info(
        "This version intentionally keeps assignments in the Streamlit session. "
        "Use the Excel export as the review/approval record. The existing Supabase "
        "tables are not modified by this application."
    )

    show_cols = [
        "Day", "Period", "Class", "Subject", "Faculty",
        "Old Room", "Proposed Room", "Allocation",
        "Shift Block", "Status", "Reason",
    ]

    st.download_button(
        "⬇️ Download Proposed Allocation CSV",
        proposed[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="semi_automatic_room_allocation.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "📊 Download Complete Excel Report",
        to_excel(
            timetable,
            proposed[show_cols],
            mapping,
            rooms,
            hurdles,
            manual_df,
        ),
        file_name="semi_automatic_room_allocation_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


st.markdown("---")
st.caption(
    "Semi-Automatic Room Allocation • Supabase read-only analysis • "
    "P1–P2 / P3–P4 / P5–P7 movement constraints preserved • "
    "AIT→B37 • DTI→Seminar hall • Library P4/P7→B41/B42 • "
    "PHY/LAC theory rooms only • B27/B37/C21 single-hour CE/AT_I/CP only"
)
