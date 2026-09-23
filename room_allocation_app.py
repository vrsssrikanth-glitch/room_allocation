import io
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from supabase import Client, create_client

# ==========================================================
# INDEPENDENT / READ-ONLY ROOM ALLOCATION APPLICATION
# ==========================================================

st.set_page_config(
    page_title="Timetable & Automatic Room Allocation",
    page_icon="🏫",
    layout="wide",
)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
PERIODS = [1, 2, 3, 4, 5, 6, 7]
SHIFT_BLOCKS = {
    1: "P1-P2", 2: "P1-P2",
    3: "P3-P4", 4: "P3-P4",
    5: "P5-P7", 6: "P5-P7", 7: "P5-P7",
}
TABLE_TIMETABLE = "timetable"
TABLE_ROOMS = "rooms"
TABLE_LABS = "labs"
TABLE_FACULTY = "faculty"

# Class specific room mapping rules
CLASS_ROOM_MAP = {
    "ECE": ["A41", "A42", "A45"],
    "CAI": ["A46"],
    "CSM": ["B34", "B35"],
    "IT": ["B43", "A38"],
    "CSC": ["B43", "A38"],
    "CSD": ["B43", "A38"],
    "CSE": ["B44", "B46", "B47"]
}

# Rooms excluded from fallback theory classes
EXCLUDED_THEORY_FALLBACK = {"B41", "B42"}


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
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required.")
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


def normalize_timetable(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Class", "Subject", "Faculty", "Day", "Period", "Room"])

    class_col = find_col(df, "class_id", "class")
    subject_col = find_col(df, "subject_id", "subject")
    faculty_col = find_col(df, "faculty_id", "faculty", "faculty_name")
    day_col = find_col(df, "day")
    period_col = find_col(df, "period")
    room_col = find_col(df, "room", "room_id")

    missing = []
    for label, col in [("class_id", class_col), ("subject", subject_col), ("day", day_col), ("period", period_col)]:
        if col is None:
            missing.append(label)
    if missing:
        raise ValueError("timetable table is missing required columns: " + ", ".join(missing))

    out = pd.DataFrame({
        "Class": df[class_col].map(clean),
        "Subject": df[subject_col].map(clean),
        "Faculty": df[faculty_col].map(clean) if faculty_col else "",
        "Day": df[day_col].map(clean).str.title(),
        "Period": pd.to_numeric(df[period_col], errors="coerce"),
        "Room": df[room_col].map(clean) if room_col else "",
    })
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
        raise ValueError("rooms table must contain Room_ID (or Room / Room_No / Room_Number).")
    out = pd.DataFrame({
        "Room_ID": df[room_col].map(clean),
        "Type": df[type_col].map(clean) if type_col else "",
    })
    out = out[out["Room_ID"] != ""].copy()
    out["_key"] = out["Room_ID"].map(norm)
    return out.drop_duplicates("_key", keep="first").drop(columns=["_key"]).reset_index(drop=True)


def normalize_labs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Lab_Subject", "Room"])
    subject_col = find_col(df, "lab_subject", "subject", "lab")
    room_col = find_col(df, "room", "room_id", "room_no", "room_number")
    if subject_col is None or room_col is None:
        return pd.DataFrame(columns=["Lab_Subject", "Room"])
    out = pd.DataFrame({"Lab_Subject": df[subject_col].map(clean), "Room": df[room_col].map(clean)})
    out = out[(out["Lab_Subject"] != "") & (out["Room"] != "")]
    return out.reset_index(drop=True)


def normalize_faculty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Faculty_ID", "Faculty_Name"])
    id_col = find_col(df, "faculty_id", "id")
    name_col = find_col(df, "faculty_name", "name")
    if id_col is None or name_col is None:
        return pd.DataFrame(columns=["Faculty_ID", "Faculty_Name"])
    out = pd.DataFrame({"Faculty_ID": df[id_col].map(clean), "Faculty_Name": df[name_col].map(clean)})
    out = out[(out["Faculty_ID"] != "") & (out["Faculty_Name"] != "")].copy()
    out["_key"] = out["Faculty_ID"].map(norm)
    return out.drop_duplicates("_key", keep="first").drop(columns=["_key"]).reset_index(drop=True)


def build_faculty_map(faculty: pd.DataFrame) -> Dict[str, str]:
    return {norm(r["Faculty_ID"]): clean(r["Faculty_Name"]) for _, r in faculty.iterrows()}


def build_lab_map(labs: pd.DataFrame) -> Dict[str, str]:
    return {norm(r["Lab_Subject"]): clean(r["Room"]) for _, r in labs.iterrows()}


def is_lab(subject: str, lab_map: Dict[str, str]) -> bool:
    s = norm(subject)
    return s in lab_map or "LAB" in s


def room_is_lab_type(room_type: str) -> bool:
    t = norm(room_type)
    return any(k in t for k in ["LAB", "LABORATORY", "COMPUTER", "WORKSHOP"])


def room_is_theory_type(room_type: str) -> bool:
    t = norm(room_type)
    return "THEORY" in t or t in {"CLASSROOM", "LECTURE", "LECTURE HALL"}


def shift_block(period: int) -> str:
    return SHIFT_BLOCKS[int(period)]


def get_preferred_rooms(cls_name: str) -> List[str]:
    c_upper = norm(cls_name)
    for prefix, allowed in CLASS_ROOM_MAP.items():
        if prefix in c_upper:
            return allowed
    return []


def allocate_rooms(timetable: pd.DataFrame, rooms: pd.DataFrame, labs: pd.DataFrame):
    """Deterministic allocation enforcing preferred room rotation and constraints."""
    if timetable.empty:
        return timetable.copy(), {"Timetable periods": 0}, pd.DataFrame()

    lab_map = build_lab_map(labs)
    all_rooms = [clean(x) for x in rooms["Room_ID"].tolist() if clean(x)]
    room_type = {norm(r["Room_ID"]): clean(r["Type"]) for _, r in rooms.iterrows()}
    theory_rooms = [r for r in all_rooms if room_is_theory_type(room_type.get(norm(r), ""))]
    if not theory_rooms:
        theory_rooms = all_rooms[:]

    occupancy: Dict[Tuple[str, int], set] = defaultdict(set)
    proposed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    day_order = {d: i for i, d in enumerate(DAYS)}
    ordered = timetable.copy()
    ordered["_d"] = ordered["Day"].map(day_order)
    ordered = ordered.sort_values(["_d", "Period", "Class", "Subject"]).drop(columns=["_d"])

    # Phase 1: Allocate fixed labs
    for _, row in ordered.iterrows():
        if not is_lab(row["Subject"], lab_map):
            continue
        mapped = lab_map.get(norm(row["Subject"]))
        if not mapped:
            failures.append({"Day": row["Day"], "Period": row["Period"], "Class": row["Class"], "Subject": row["Subject"], "Reason": "Lab detected but not mapped in labs table"})
            continue
        if norm(mapped) not in {norm(x) for x in all_rooms}:
            failures.append({"Day": row["Day"], "Period": row["Period"], "Class": row["Class"], "Subject": row["Subject"], "Reason": f"Lab room {mapped} is not present in rooms table"})
            continue
        key = (row["Day"], int(row["Period"]))
        if norm(mapped) in {norm(x) for x in occupancy[key]}:
            failures.append({"Day": row["Day"], "Period": row["Period"], "Class": row["Class"], "Subject": row["Subject"], "Reason": f"Fixed lab room clash: {mapped}"})
            continue
        occupancy[key].add(mapped)
        proposed.append({
            "Day": row["Day"], "Period": int(row["Period"]), "Class": row["Class"], "Subject": row["Subject"],
            "Faculty": row["Faculty"], "Old Room": row["Room"], "Proposed Room": mapped,
            "Allocation": "LAB-FIXED", "Shift Block": "LAB", "Status": "LAB FIXED", "Reason": "",
        })

    # Phase 2: Theory Room Allocation
    theory = ordered[~ordered["Subject"].map(lambda x: is_lab(x, lab_map))].copy()
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for _, row in theory.iterrows():
        groups[(row["Day"], row["Class"], shift_block(int(row["Period"])))].append(row.to_dict())

    group_items = sorted(groups.items(), key=lambda kv: (-len(kv[1]), day_order[kv[0][0]], kv[0][2], kv[0][1]))
    last_room: Dict[Tuple[str, str], str] = {}
    class_day_rooms: Dict[Tuple[str, str], set] = defaultdict(set)

    for (day, cls, block), group in group_items:
        periods = sorted({int(x["Period"]) for x in group})
        preferred = get_preferred_rooms(cls)
        
        # Build priority candidates for room allocation
        candidates = []
        candidate_pool = preferred + [r for r in theory_rooms if r not in preferred and norm(r) not in EXCLUDED_THEORY_FALLBACK]

        for idx, room in enumerate(candidate_pool):
            rk = norm(room)
            if any(rk in {norm(x) for x in occupancy[(day, p)]} for p in periods):
                continue
            score = 0
            # Higher priority for section room mapping preferences
            if room in preferred:
                score += 1000000 - (preferred.index(room) * 1000)
            prev = last_room.get((cls, day), "")
            if prev and rk == norm(prev):
                score += 50000
            if rk in {norm(x) for x in class_day_rooms[(cls, day)]}:
                score += 15000
            score -= idx
            candidates.append((score, room))

        if not candidates:
            for x in group:
                proposed.append({
                    "Day": day, "Period": int(x["Period"]), "Class": cls, "Subject": x["Subject"],
                    "Faculty": x.get("Faculty", ""), "Old Room": x.get("Room", ""), "Proposed Room": "UNALLOCATED",
                    "Allocation": "FAILED", "Shift Block": block, "Status": "FAILED", "Reason": "No room is free for the shift block",
                })
            continue

        selected_room = max(candidates, key=lambda z: z[0])[1]
        last_room[(cls, day)] = selected_room
        class_day_rooms[(cls, day)].add(selected_room)

        for x in group:
            p = int(x["Period"])
            occupancy[(day, p)].add(selected_room)
            old = clean(x.get("Room", ""))
            status = "UNCHANGED" if old and norm(old) == norm(selected_room) else "ROOM CHANGE"
            proposed.append({
                "Day": day, "Period": p, "Class": cls, "Subject": x["Subject"], "Faculty": x.get("Faculty", ""),
                "Old Room": old, "Proposed Room": selected_room, "Allocation": "THEORY-AUTO", "Shift Block": block,
                "Status": status, "Reason": "",
            })

    result = pd.DataFrame(proposed)
    if result.empty:
        result = pd.DataFrame(columns=["Day", "Period", "Class", "Subject", "Faculty", "Old Room", "Proposed Room", "Allocation", "Shift Block", "Status", "Reason"])
    result["_d"] = result["Day"].map(day_order)
    result = result.sort_values(["_d", "Class", "Period", "Subject"]).drop(columns=["_d"]).reset_index(drop=True)

    total = len(timetable)
    allocated = int((result["Proposed Room"] != "UNALLOCATED").sum()) if not result.empty else 0
    lab_count = int((result["Allocation"] == "LAB-FIXED").sum()) if not result.empty else 0
    theory_count = int((result["Allocation"] == "THEORY-AUTO").sum()) if not result.empty else 0
    changes = int((result["Status"] == "ROOM CHANGE").sum()) if not result.empty else 0

    metrics = {
        "Timetable periods": total,
        "Allocated": allocated,
        "Unallocated": total - allocated,
        "Lab periods fixed": lab_count,
        "Theory periods": theory_count,
        "Room changes from current timetable": changes,
        "Rooms in Supabase": len(all_rooms),
        "Allocation %": round((allocated / total) * 100, 2) if total else 0.0,
    }
    return result, metrics, pd.DataFrame(failures)


def faculty_display(value: str, faculty_map: Dict[str, str]) -> str:
    v = clean(value)
    if not v:
        return "—"
    return faculty_map.get(norm(v), v)


def class_timetable_styled_grid(result: pd.DataFrame, class_name: str, faculty_map: Dict[str, str]) -> pd.DataFrame:
    """Returns styled HTML content for clean colored visibility."""
    g = result[result["Class"] == class_name].copy()
    rows = []
    for day in DAYS:
        row: Dict[str, str] = {"Day": day}
        for p in PERIODS:
            x = g[(g["Day"] == day) & (g["Period"] == p)]
            if x.empty:
                row[f"P{p}"] = "<div style='color:#888; text-align:center;'>—</div>"
                continue
            r = x.iloc[0]
            room = clean(r["Proposed Room"])
            subject = clean(r["Subject"]) or "—"
            faculty = faculty_display(r["Faculty"], faculty_map)

            if r["Allocation"] == "LAB-FIXED":
                bg = "#e3f2fd"
                border = "#2196f3"
                room_txt = f"<b>Room: {room} [LAB]</b>"
            elif room == "UNALLOCATED":
                bg = "#ffebee"
                border = "#f44336"
                room_txt = "<b>UNALLOCATED</b>"
            else:
                bg = "#f1f8e9"
                border = "#8bc34a"
                room_txt = f"<b>Room: {room}</b>"

            cell_html = f"""
            <div style="background-color:{bg}; border-left:4px solid {border}; padding:6px; border-radius:4px;">
                <div style="font-weight:bold; color:#0d47a1; font-size:13px;">{subject}</div>
                <div style="color:#333; font-size:11px;">👤 {faculty}</div>
                <div style="color:#2e7d32; font-size:12px; margin-top:2px;">{room_txt}</div>
            </div>
            """
            row[f"P{p}"] = cell_html
        rows.append(row)
    return pd.DataFrame(rows)


def vacancy_theory_grid(result: pd.DataFrame, rooms: pd.DataFrame) -> pd.DataFrame:
    """Generates vacancy matrix strictly for theory rooms."""
    room_type_map = {norm(r["Room_ID"]): clean(r["Type"]) for _, r in rooms.iterrows()}
    theory_rooms = sorted([
        clean(x) for x in rooms["Room_ID"].tolist() 
        if clean(x) and room_is_theory_type(room_type_map.get(norm(x), ""))
    ], key=lambda x: x.upper())

    rows = []
    for day in DAYS:
        row = {"Day": day}
        for p in PERIODS:
            used = {
                norm(x)
                for x in result.loc[
                    (result["Day"] == day) & (result["Period"] == p) & (result["Proposed Room"] != "UNALLOCATED"),
                    "Proposed Room",
                ]
            }
            vacant = [r for r in theory_rooms if norm(r) not in used]
            row[f"P{p}"] = ", ".join(vacant) if vacant else "— NONE —"
        rows.append(row)
    return pd.DataFrame(rows)


def room_occupancy_summary(result: pd.DataFrame, rooms: pd.DataFrame) -> pd.DataFrame:
    room_ids = sorted([clean(x) for x in rooms["Room_ID"].tolist() if clean(x)], key=lambda x: x.upper())
    total_slots = len(DAYS) * len(PERIODS)
    rows = []
    for room in room_ids:
        used = result[(result["Proposed Room"].map(norm) == norm(room)) & (result["Proposed Room"] != "UNALLOCATED")]
        occupied = int(used[["Day", "Period"]].drop_duplicates().shape[0])
        vacant = total_slots - occupied
        occ_pct = (occupied / total_slots * 100) if total_slots else 0.0
        rows.append({
            "Room": room,
            "Type": rooms.loc[rooms["Room_ID"].map(norm) == norm(room), "Type"].iloc[0] if not rooms.loc[rooms["Room_ID"].map(norm) == norm(room)].empty else "",
            "Occupied periods": occupied,
            "Vacant periods": vacant,
            "Occupancy %": round(occ_pct, 2),
            "Vacancy %": round(100 - occ_pct, 2),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["Occupancy %", "Room"], ascending=[False, True]).reset_index(drop=True)


def overall_occupancy(result: pd.DataFrame, rooms: pd.DataFrame) -> Dict[str, float]:
    total_slots = len(rooms) * len(DAYS) * len(PERIODS)
    occupied_slots = int(result[result["Proposed Room"] != "UNALLOCATED"][["Day", "Period", "Proposed Room"]].drop_duplicates().shape[0]) if not result.empty else 0
    vacancy_slots = max(total_slots - occupied_slots, 0)
    pct = (occupied_slots / total_slots * 100) if total_slots else 0.0
    return {
        "Occupied room-period slots": occupied_slots,
        "Vacant room-period slots": vacancy_slots,
        "Total room-period slots": total_slots,
        "Overall occupancy %": round(pct, 2),
        "Overall vacancy %": round(100 - pct, 2),
    }


# ==========================================================
# STREAMLIT UI
# ==========================================================
st.title("🏫 Timetable & Automatic Room Allocation")
st.caption("Read-only allocation enforcing specific section-to-room mapping rules.")

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reload Supabase data", use_container_width=True):
        fetch_table.clear()
        st.rerun()
    st.markdown("### Department Room Mapping")
    st.write("• **ECE 1-4**: A41, A42, A45")
    st.write("• **CAI 1-2**: A46")
    st.write("• **CSM 1-3**: B34, B35")
    st.write("• **IT, CSC, CSD**: B43, A38")
    st.write("• **CSE 1-4**: B44, B46, B47")
    st.write("• **B41, B42**: Labs excluded from theory fallback")

try:
    with st.spinner("Fetching data from Supabase..."):
        raw_timetable = fetch_table(TABLE_TIMETABLE)
        raw_rooms = fetch_table(TABLE_ROOMS)
        raw_labs = fetch_table(TABLE_LABS)
        raw_faculty = fetch_table(TABLE_FACULTY)
        timetable = normalize_timetable(raw_timetable)
        rooms = normalize_rooms(raw_rooms)
        labs = normalize_labs(raw_labs)
        faculty = normalize_faculty(raw_faculty)
        faculty_map = build_faculty_map(faculty)
except Exception as exc:
    st.error(f"Could not read Supabase data: {exc}")
    st.stop()

if timetable.empty or rooms.empty:
    st.error("Missing timetable or rooms data in Supabase.")
    st.stop()

proposed, metrics, failures = allocate_rooms(timetable, rooms, labs)

overall = overall_occupancy(proposed, rooms)
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total Periods", metrics["Timetable periods"])
m2.metric("Allocated", metrics["Allocated"])
m3.metric("Unallocated", metrics["Unallocated"])
m4.metric("Rooms", metrics["Rooms in Supabase"])
m5.metric("Occupancy", f"{overall['Overall occupancy %']}%")
m6.metric("Vacancy", f"{overall['Overall vacancy %']}%")

classes = sorted(timetable["Class"].unique().tolist())
selected_class = st.selectbox("Select class to display timetable", classes)

# 1. VISUAL CLASS TIMETABLE (Styled HTML for clear colors)
st.header(f"1. Standard Timetable – {selected_class}")
styled_grid = class_timetable_styled_grid(proposed, selected_class, faculty_map)
st.write(styled_grid.to_html(escape=False, index=False), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. VACANT THEORY ROOMS
st.header("2. Vacant Theory Rooms Throughout Week")
st.caption("Excludes lab rooms. Displays only available theory rooms per period slot.")
theory_vacancies = vacancy_theory_grid(proposed, rooms)
st.dataframe(theory_vacancies, use_container_width=True, hide_index=True)

# 3. ROOM OCCUPANCY %
st.header("3. Overall Room Occupancy Summary")
occupancy = room_occupancy_summary(proposed, rooms)
st.dataframe(occupancy, use_container_width=True, hide_index=True)
