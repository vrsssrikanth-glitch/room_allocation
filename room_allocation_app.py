import io
import os
import zlib
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from supabase import Client, create_client

# ==========================================================
# STREAMLIT PAGE CONFIG & CARD CSS STYLING
# ==========================================================

st.set_page_config(
    page_title="Timetable & Room Allocation System",
    page_icon="🏫",
    layout="wide",
)

st.markdown("""
    <style>
    .main {
        background-color: #FAFAFA;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1E293B;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.02);
    }

    /* Timetable Grid & Cards Styling */
    .tt-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 6px;
        margin-top: 10px;
    }
    .tt-table th {
        background-color: #FAFAFA;
        color: #333333;
        font-weight: bold;
        text-align: center;
        padding: 8px;
        font-size: 15px;
    }
    .tt-table td {
        vertical-align: middle;
        padding: 2px;
    }
    .tt-day-label {
        font-weight: 600;
        color: #212529;
        font-size: 14px;
        padding: 8px 12px !important;
        white-space: nowrap;
    }
    .tt-card {
        border-radius: 6px;
        padding: 10px 6px;
        text-align: center;
        min-height: 58px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .tt-subject {
        font-weight: 700;
        font-size: 12px;
        line-height: 1.2;
        margin-bottom: 2px;
    }
    .tt-faculty {
        font-weight: 600;
        font-size: 11px;
        line-height: 1.2;
    }
    .tt-room {
        font-weight: 500;
        font-size: 10px;
        opacity: 0.85;
        margin-top: 2px;
    }

    /* Color Palette Themes */
    .theme-teal {
        background-color: #E6F8F3;
        border-left: 5px solid #00B074;
    }
    .theme-teal .tt-subject, .theme-teal .tt-faculty, .theme-teal .tt-room { color: #007A50; }

    .theme-pink {
        background-color: #FCE8F3;
        border-left: 5px solid #D63384;
    }
    .theme-pink .tt-subject, .theme-pink .tt-faculty, .theme-pink .tt-room { color: #8A1551; }

    .theme-lavender {
        background-color: #ECEAFB;
        border-left: 5px solid #4B38B3;
    }
    .theme-lavender .tt-subject, .theme-lavender .tt-faculty, .theme-lavender .tt-room { color: #2B1883; }

    .theme-yellow {
        background-color: #F7FBE2;
        border-left: 5px solid #84A900;
    }
    .theme-yellow .tt-subject, .theme-yellow .tt-faculty, .theme-yellow .tt-room { color: #4B6100; }

    .theme-empty {
        background-color: #F8F9FA;
        border: 1px dashed #DEE2E6;
        color: #ADB5BD;
    }
    </style>
""", unsafe_allow_html=True)

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

SPECIAL_ACTIVITY_ROOMS = ["B41", "B42"]


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
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY environment variables are required.")
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
    name_col = find_col(df, "name", "faculty_name")
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


def is_special_activity(subject: str) -> bool:
    s = norm(subject)
    keywords = ["MAKER", "MAKERS", "LIB", "LIBRARY", "NEWS", "WEEKLY TEST", "TEST"]
    return any(kw in s for kw in keywords)


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
    all_supabase_rooms = [clean(r) for r in rooms["Room_ID"].unique() if clean(r)]
    theory_rooms = [
        clean(r["Room_ID"]) for _, r in rooms.iterrows() 
        if clean(r["Room_ID"]) and norm(r["Type"]) in ["THEORY", "CLASSROOM", "LECTURE", ""]
    ]
    if not theory_rooms:
        theory_rooms = all_supabase_rooms[:]

    occupancy: Dict[Tuple[str, int], set] = defaultdict(set)
    proposed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    day_order = {d: i for i, d in enumerate(DAYS)}
    ordered = timetable.copy()
    ordered["_d"] = ordered["Day"].map(day_order)
    ordered = ordered.sort_values(["_d", "Period", "Class", "Subject"]).drop(columns=["_d"])

    # 1. Special Activity Allocation & Rule for EEE-2 Weekly Test
    special_df = ordered[ordered["Subject"].map(is_special_activity)].copy()
    for _, row in special_df.iterrows():
        key = (row["Day"], int(row["Period"]))
        cls_norm = norm(row["Class"])
        subj_norm = norm(row["Subject"])
        assigned_room = None

        # Custom Rule: EEE-2 Weekly Test -> Forced Room B41
        if ("EEE-2" in cls_norm or "EEE 2" in cls_norm) and "WEEKLY TEST" in subj_norm:
            if "B41" not in {norm(x) for x in occupancy[key]}:
                assigned_room = "B41"

        if not assigned_room:
            for rm in SPECIAL_ACTIVITY_ROOMS:
                if norm(rm) not in {norm(x) for x in occupancy[key]}:
                    assigned_room = rm
                    break
        
        if assigned_room:
            occupancy[key].add(assigned_room)
            proposed.append({
                "Day": row["Day"], "Period": int(row["Period"]), "Class": row["Class"], "Subject": row["Subject"],
                "Faculty": row["Faculty"], "Old Room": row["Room"], "Proposed Room": assigned_room,
                "Allocation": "SPECIAL-ACTIVITY", "Shift Block": "SPECIAL", "Status": "SPECIAL ASSIGNED", "Reason": "",
            })
        else:
            failures.append({"Day": row["Day"], "Period": row["Period"], "Class": row["Class"], "Subject": row["Subject"], "Reason": "Special room occupied"})

    # 2. Fixed Lab Allocation
    labs_df = ordered[ordered["Subject"].map(lambda x: is_lab(x, lab_map) and not is_special_activity(x))].copy()
    for _, row in labs_df.iterrows():
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

    # 3. Theory Room Allocation with Monday C21 Addition
    theory = ordered[~ordered["Subject"].map(lambda x: is_lab(x, lab_map) or is_special_activity(x))].copy()
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for _, row in theory.iterrows():
        groups[(row["Day"], row["Class"], shift_block(int(row["Period"])))].append(row.to_dict())

    group_items = sorted(groups.items(), key=lambda kv: (-len(kv[1]), day_order[kv[0][0]], kv[0][2], kv[0][1]))
    last_room: Dict[Tuple[str, str], str] = {}
    class_day_rooms: Dict[Tuple[str, str], set] = defaultdict(set)

    for (day, cls, block), group in group_items:
        periods = sorted({int(x["Period"]) for x in group})
        preferred = get_preferred_rooms(cls)
        
        all_day_additions = ["B37", "B27"]
        saturday_additions = ["B02"] if day.upper() == "SATURDAY" else []
        # Rule: Allow C21 for Monday slots
        monday_additions = ["C21"] if day.upper() == "MONDAY" else []
        
        active_extra_rooms = all_day_additions + saturday_additions + monday_additions
        
        candidate_pool = preferred + active_extra_rooms + [
            r for r in theory_rooms if r not in preferred and r not in active_extra_rooms
        ]

        candidates = []
        for idx, room in enumerate(candidate_pool):
            rk = norm(room)
            if any(rk in {norm(x) for x in occupancy[(day, p)]} for p in periods):
                continue
            
            score = 0
            if room in preferred:
                score += 1000000 - (preferred.index(room) * 1000)
            elif room in active_extra_rooms:
                score += 500000 - (active_extra_rooms.index(room) * 1000)
            else:
                score += 10000 - (idx * 10)
                
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


def faculty_display(value: str, faculty_map: Dict[str, str]) -> str:
    v = clean(value)
    if not v:
        return ""
    return faculty_map.get(norm(v), v)


def get_subject_color_theme(subject: str) -> str:
    s = norm(subject)
    if "TEST" in s or "CP" in s or "NSS" in s:
        return "theme-teal"
    elif "LAC" in s or "AIT" in s or "LAB" in s:
        return "theme-pink"
    elif "CE" in s or "PHY" in s:
        return "theme-lavender"
    elif "AI_T" in s or "LIB" in s:
        return "theme-yellow"
    return "theme-teal"


def render_html_timetable(result: pd.DataFrame, class_name: str, faculty_map: Dict[str, str]):
    g = result[result["Class"] == class_name].copy()
    
    html = '<table class="tt-table"><thead><tr><th>Day</th>'
    for p in PERIODS:
        html += f'<th>P{p}</th>'
    html += '</tr></thead><tbody>'

    for day in DAYS:
        html += f'<tr><td class="tt-day-label">{day}</td>'
        for p in PERIODS:
            x = g[(g["Day"] == day) & (g["Period"] == p)]
            if x.empty:
                html += '<td><div class="tt-card theme-empty"><span class="tt-subject">—</span></div></td>'
            else:
                r = x.iloc[0]
                subj = clean(r["Subject"]) or "—"
                fac = faculty_display(r["Faculty"], faculty_map)
                room = clean(r["Proposed Room"])
                fac_str = f"({fac})" if fac else ""
                room_str = f"[{room}]" if room else ""
                
                theme_class = get_subject_color_theme(subj)

                html += f'''
                <td>
                    <div class="tt-card {theme_class}">
                        <div class="tt-subject">{subj}</div>
                        <div class="tt-faculty">{fac_str}</div>
                        <div class="tt-room">{room_str}</div>
                    </div>
                </td>
                '''
        html += '</tr>'
    html += '</tbody></table>'
    
    st.markdown(html, unsafe_allow_html=True)


# ==========================================================
# APP EXECUTION & UI
# ==========================================================
st.title("🏫 Timetable & Automatic Room Allocation")
st.caption("Powered by Streamlit & Supabase")

with st.sidebar:
    st.header("⚙️ Administrative Controls")
    if st.button("🔄 Refresh Data", use_container_width=True):
        fetch_table.clear()
        st.rerun()

try:
    with st.spinner("Connecting to Supabase..."):
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
    st.error(f"Error loading database: {exc}")
    st.stop()

if timetable.empty or rooms.empty:
    st.error("Timetable or rooms table is empty.")
    st.stop()

proposed, metrics, failures = allocate_rooms(timetable, rooms, labs)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Periods", metrics["Timetable periods"])
m2.metric("Allocated Periods", metrics["Allocated"])
m3.metric("Unallocated Slots", metrics["Unallocated"])
m4.metric("Allocation Rate", f"{metrics['Allocation %']}%")

st.markdown("<br>", unsafe_allow_html=True)

classes = sorted(timetable["Class"].unique().tolist())
selected_class = st.selectbox("📌 Select Class / Section", classes)

coord_info = coordinator_map.get(norm(selected_class))
if coord_info:
    st.info(f"👤 **Class Coordinator:** {coord_info['name']} &nbsp;|&nbsp; 📱 **Mobile:** {coord_info['mobile']}")

st.subheader(f"📅 Schedule – {selected_class}")
render_html_timetable(proposed, selected_class, faculty_map)
