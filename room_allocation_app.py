import io
import os
import zlib
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

# Preferred Room Mapping
CLASS_ROOM_MAP = {
    "ECE": ["A41", "A42", "A45"],
    "CSE": ["B44", "B46", "B47"],
    "CSM": ["A46", "A47"],
    "CAI": ["B43", "B35", "B36", "B34", "A48", "A38"],
    "IT":  ["B43", "B35", "B36", "B34", "A48", "A38"],
    "CSC": ["B43", "B35", "B36", "B34", "A48", "A38"],
    "CSD": ["B43", "B35", "B36", "B34", "A48", "A38"],
    "EEE": ["B43", "B35", "B36", "B34", "A48", "A38"],
}

# General Fallback Rooms
GENERAL_FALLBACK_ROOMS = ["B37", "B27"]

COLOR_PALETTE = [
    "#E3F2FD", "#F3E5F5", "#E8F5E9", "#FFF3E0", 
    "#FCE4EC", "#E0F7FA", "#FFFDE7", "#F3E5F5"
]


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
        raise ValueError("rooms table must contain Room_ID.")
    out = pd.DataFrame({
        "Room_ID": df[room_col].map(clean),
        "Type": df[type_col].map(clean) if type_col else "Theory",
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
        return pd.DataFrame(columns=["Faculty_ID", "Faculty_Name", "Mobile", "Is_Coordinator", "Class_Coordinated"])
    id_col = find_col(df, "faculty_id", "id")
    name_col = find_col(df, "faculty_name", "name")
    mobile_col = find_col(df, "mobile", "phone", "contact")
    coord_col = find_col(df, "is_coordinator", "coordinator")
    class_coord_col = find_col(df, "class_coordinated", "class_coordinator")

    out = pd.DataFrame({
        "Faculty_ID": df[id_col].map(clean) if id_col else df[name_col].map(clean),
        "Faculty_Name": df[name_col].map(clean) if name_col else "",
        "Mobile": df[mobile_col].map(clean) if mobile_col else "",
        "Is_Coordinator": df[coord_col].map(clean) if coord_col else "",
        "Class_Coordinated": df[class_coord_col].map(clean) if class_coord_col else "",
    })
    return out[out["Faculty_Name"] != ""].reset_index(drop=True)


def build_faculty_map(faculty: pd.DataFrame) -> Dict[str, str]:
    return {norm(r["Faculty_ID"]): clean(r["Faculty_Name"]) for _, r in faculty.iterrows()}


def build_coordinator_map(faculty: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    coord_map = {}
    for _, r in faculty.iterrows():
        is_coord = norm(r["Is_Coordinator"])
        class_coord = clean(r["Class_Coordinated"])
        if is_coord in ["YES", "TRUE", "1"] and class_coord:
            coord_map[norm(class_coord)] = {
                "name": clean(r["Faculty_Name"]),
                "mobile": clean(r["Mobile"]),
            }
    return coord_map


def build_lab_map(labs: pd.DataFrame) -> Dict[str, str]:
    return {norm(r["Lab_Subject"]): clean(r["Room"]) for _, r in labs.iterrows()}


def is_lab(subject: str, lab_map: Dict[str, str]) -> bool:
    s = norm(subject)
    return s in lab_map or "LAB" in s


def shift_block(period: int) -> str:
    return SHIFT_BLOCKS[int(period)]


def get_preferred_rooms(cls_name: str) -> List[str]:
    c_upper = norm(cls_name)
    for prefix, allowed in CLASS_ROOM_MAP.items():
        if prefix in c_upper:
            return allowed
    return []


def allocate_rooms(timetable: pd.DataFrame, rooms: pd.DataFrame, labs: pd.DataFrame):
    if timetable.empty:
        return timetable.copy(), {"Timetable periods": 0}, pd.DataFrame()

    lab_map = build_lab_map(labs)
    all_rooms = [clean(r) for r in rooms["Room_ID"].unique() if clean(r)]
    
    occupancy: Dict[Tuple[str, int], set] = defaultdict(set)
    proposed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    day_order = {d: i for i, d in enumerate(DAYS)}
    ordered = timetable.copy()
    ordered["_d"] = ordered["Day"].map(day_order)
    ordered = ordered.sort_values(["_d", "Period", "Class", "Subject"]).drop(columns=["_d"])

    # 1. Fixed Lab Allocation
    for _, row in ordered.iterrows():
        if not is_lab(row["Subject"], lab_map):
            continue
        mapped = lab_map.get(norm(row["Subject"]))
        if not mapped:
            failures.append({"Day": row["Day"], "Period": row["Period"], "Class": row["Class"], "Subject": row["Subject"], "Reason": "Lab missing in labs table"})
            continue
        key = (row["Day"], int(row["Period"]))
        if norm(mapped) in {norm(x) for x in occupancy[key]}:
            failures.append({"Day": row["Day"], "Period": row["Period"], "Class": row["Class"], "Subject": row["Subject"], "Reason": f"Fixed lab clash: {mapped}"})
            continue
        occupancy[key].add(mapped)
        proposed.append({
            "Day": row["Day"], "Period": int(row["Period"]), "Class": row["Class"], "Subject": row["Subject"],
            "Faculty": row["Faculty"], "Old Room": row["Room"], "Proposed Room": mapped,
            "Allocation": "LAB-FIXED", "Shift Block": "LAB", "Status": "LAB FIXED", "Reason": "",
        })

    # 2. Theory Room Allocation with Pending Rooms Filling Fallback
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
        
        # Room search sequence: Preferred -> General Fallbacks -> Remaining Database Rooms
        day_pool = preferred + GENERAL_FALLBACK_ROOMS[:]
        if day == "Saturday":
            day_pool.append("B02")
        
        # Include any remaining rooms in database as ultimate fallback
        for extra_r in all_rooms:
            if extra_r not in day_pool:
                day_pool.append(extra_r)

        candidates = []
        for idx, room in enumerate(day_pool):
            rk = norm(room)
            if any(rk in {norm(x) for x in occupancy[(day, p)]} for p in periods):
                continue
            
            score = 0
            if room in preferred:
                score += 1000000 - (preferred.index(room) * 1000)
            elif room in GENERAL_FALLBACK_ROOMS or room == "B02":
                score += 100000 - (idx * 500)
            else:
                score += 10000 - (idx * 100) # Standard database fallback
                
            prev = last_room.get((cls, day), "")
            if prev and rk == norm(prev):
                score += 50000
            if rk in {norm(x) for x in class_day_rooms[(cls, day)]}:
                score += 15000
            
            candidates.append((score, room))

        if not candidates:
            for x in group:
                proposed.append({
                    "Day": day, "Period": int(x["Period"]), "Class": cls, "Subject": x["Subject"],
                    "Faculty": x.get("Faculty", ""), "Old Room": x.get("Room", ""), "Proposed Room": "UNALLOCATED",
                    "Allocation": "FAILED", "Shift Block": block, "Status": "FAILED", "Reason": "No available room in shift block",
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
    metrics = {
        "Timetable periods": total,
        "Allocated": allocated,
        "Unallocated": total - allocated,
        "Allocation %": round((allocated / total) * 100, 2) if total else 0.0,
    }
    return result, metrics, pd.DataFrame(failures)


def vacancy_theory_grid(result: pd.DataFrame, rooms: pd.DataFrame, selected_day: str) -> pd.DataFrame:
    all_rooms = sorted([clean(r) for r in rooms["Room_ID"].unique() if clean(r)])
    rows = []
    
    for p in PERIODS:
        occupied_rooms = set(
            result[(result["Day"] == selected_day) & (result["Period"] == p) & (result["Proposed Room"] != "UNALLOCATED")]["Proposed Room"].map(norm)
        )
        vacant = [r for r in all_rooms if norm(r) not in occupied_rooms]
        rows.append({
            "Period": f"Period {p}",
            "Vacant Rooms Count": len(vacant),
            "Available Vacant Rooms": ", ".join(vacant) if vacant else "— NONE (ALL OCCUPIED) —"
        })
    return pd.DataFrame(rows)


def room_class_occupancy_grid(result: pd.DataFrame, rooms: pd.DataFrame, selected_day: str) -> pd.DataFrame:
    all_rooms = sorted([clean(r) for r in rooms["Room_ID"].unique() if clean(r)])
    rows = []
    
    for room in all_rooms:
        row = {"Room": room}
        for p in PERIODS:
            match = result[(result["Day"] == selected_day) & (result["Period"] == p) & (result["Proposed Room"] == room)]
            if not match.empty:
                c_name = match.iloc[0]["Class"]
                subj = match.iloc[0]["Subject"]
                row[f"P{p}"] = f"{c_name}\n({subj})"
            else:
                row[f"P{p}"] = "— VACANT —"
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_occupancy(result: pd.DataFrame, rooms: pd.DataFrame) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    if result.empty:
        return 0.0, pd.DataFrame(), pd.DataFrame()

    all_rooms = [clean(r) for r in rooms["Room_ID"].unique() if clean(r)]
    total_slots_per_day = len(all_rooms) * len(PERIODS)
    
    # 1. Per-Day Occupancy Rate
    day_stats = []
    for day in DAYS:
        allocated_in_day = len(result[(result["Day"] == day) & (result["Proposed Room"] != "UNALLOCATED")])
        rate = round((allocated_in_day / total_slots_per_day) * 100, 2) if total_slots_per_day else 0.0
        day_stats.append({"Day": day, "Allocated Slots": allocated_in_day, "Total Slot Capacity": total_slots_per_day, "Occupancy Rate (%)": f"{rate}%"})

    day_df = pd.DataFrame(day_stats)

    # 2. Room-Wise Utilization Statistics
    room_stats = []
    total_week_slots = len(DAYS) * len(PERIODS)
    for rm in all_rooms:
        used_slots = len(result[result["Proposed Room"] == rm])
        utilization = round((used_slots / total_week_slots) * 100, 2)
        room_stats.append({"Room": rm, "Weekly Used Slots": used_slots, "Capacity (Slots)": total_week_slots, "Utilization (%)": f"{utilization}%"})

    room_df = pd.DataFrame(room_stats).sort_values("Weekly Used Slots", ascending=False)

    # Overall Occupancy Rate
    total_capacity = total_slots_per_day * len(DAYS)
    total_allocated = len(result[result["Proposed Room"] != "UNALLOCATED"])
    overall_rate = round((total_allocated / total_capacity) * 100, 2) if total_capacity else 0.0

    return overall_rate, day_df, room_df


def faculty_display(value: str, faculty_map: Dict[str, str]) -> str:
    v = clean(value)
    if not v:
        return "—"
    return faculty_map.get(norm(v), v)


def get_subject_color(subject_str: str) -> str:
    if not subject_str or subject_str == "—":
        return "#FFFFFF"
    subj_clean = subject_str.split("\n")[0].strip().upper()
    idx = zlib.crc32(subj_clean.encode()) % len(COLOR_PALETTE)
    return COLOR_PALETTE[idx]


def class_timetable_grid(result: pd.DataFrame, class_name: str, faculty_map: Dict[str, str]) -> pd.DataFrame:
    g = result[result["Class"] == class_name].copy()
    rows = []
    for day in DAYS:
        row: Dict[str, str] = {"Day": day}
        for p in PERIODS:
            x = g[(g["Day"] == day) & (g["Period"] == p)]
            if x.empty:
                row[f"P{p}"] = "—"
                continue
            r = x.iloc[0]
            room = clean(r["Proposed Room"])
            subject = clean(r["Subject"]) or "—"
            faculty = faculty_display(r["Faculty"], faculty_map)

            if r["Allocation"] == "LAB-FIXED":
                room_txt = f"Room: {room} [LAB]"
            elif room == "UNALLOCATED":
                room_txt = "Room: UNALLOCATED"
            else:
                room_txt = f"Room: {room}"

            row[f"P{p}"] = f"{subject}\n{faculty}\n{room_txt}"
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================
# APP EXECUTION & UI
# ==========================================================
st.title("🏫 Standard Timetable & Room Allocation")

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reload Supabase data", use_container_width=True):
        fetch_table.clear()
        st.rerun()

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
        coordinator_map = build_coordinator_map(faculty)
except Exception as exc:
    st.error(f"Could not read Supabase data: {exc}")
    st.stop()

if timetable.empty or rooms.empty:
    st.error("Timetable or rooms table is empty.")
    st.stop()

proposed, metrics, failures = allocate_rooms(timetable, rooms, labs)
overall_occ, day_occ_df, room_occ_df = calculate_occupancy(proposed, rooms)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Periods", metrics["Timetable periods"])
m2.metric("Allocated Periods", metrics["Allocated"])
m3.metric("Unallocated Slots", metrics["Unallocated"])
m4.metric("Overall Room Occupancy", f"{overall_occ}%")

classes = sorted(timetable["Class"].unique().tolist())
selected_class = st.selectbox("Select Class / Section", classes)

# Class Coordinator Metadata
coord_info = coordinator_map.get(norm(selected_class))
if coord_info:
    st.info(f"📋 **Class Coordinator:** {coord_info['name']} | 📱 **Mobile:** {coord_info['mobile']}")
else:
    st.warning(f"No Class Coordinator assigned to {selected_class} in the Faculty table.")

# 1. Class Timetable
st.header(f"Standard Timetable – {selected_class}")
grid_df = class_timetable_grid(proposed, selected_class, faculty_map)

styled_df = grid_df.style.map(
    lambda val: f"background-color: {get_subject_color(val)}; font-weight: 500;",
    subset=[f"P{p}" for p in PERIODS]
)
st.dataframe(styled_df, use_container_width=True, hide_index=True, height=380)

# 2. Detailed Room Occupancy & Vacancy Analysis
st.header("🏢 Detailed Room Occupancy & Vacancy Grid")
day_filter = st.selectbox("Select Day to View Room Matrix & Vacancy", DAYS)

tab_room1, tab_room2 = st.tabs(["Class Occupancy in Rooms Matrix", "Vacant Rooms per Period Grid"])

with tab_room1:
    st.subheader(f"Room-wise Class Occupancy ({day_filter})")
    room_matrix = room_class_occupancy_grid(proposed, rooms, day_filter)
    st.dataframe(room_matrix, use_container_width=True, hide_index=True)

with tab_room2:
    st.subheader(f"Vacant Rooms Summary ({day_filter})")
    vacant_matrix = vacancy_theory_grid(proposed, rooms, day_filter)
    st.dataframe(vacant_matrix, use_container_width=True, hide_index=True)

# 3. Overall Occupancy Rate Analytics
st.header("📊 Room Occupancy Rate Summaries")
tab1, tab2 = st.tabs(["Daily Occupancy Rates", "Room-Wise Utilization Rate"])

with tab1:
    st.dataframe(day_occ_df, use_container_width=True, hide_index=True)

with tab2:
    st.dataframe(room_occ_df, use_container_width=True, hide_index=True)

# 4. Unallocated Slots Summary
unallocated_df = proposed[proposed["Proposed Room"] == "UNALLOCATED"].copy()
st.header("⚠️ Unallocated Slots Summary")

if not unallocated_df.empty:
    st.warning(f"There are {len(unallocated_df)} unallocated class slots requiring attention:")
    st.dataframe(
        unallocated_df[["Day", "Period", "Class", "Subject", "Faculty", "Reason"]], 
        use_container_width=True, 
        hide_index=True
    )
else:
    st.success("All timetable slots are successfully allocated to rooms!")
