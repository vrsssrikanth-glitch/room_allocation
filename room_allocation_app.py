import io
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from supabase import Client, create_client

# ==========================================================
# PAGE CONFIGURATION & STYLING
# ==========================================================
st.set_page_config(
    page_title="Timetable & Automatic Room Allocation",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced visuals and color palette
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f5 100%);
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .coordinator-badge {
        background-color: #e7f5ff;
        border-left: 4px solid #1c7ed6;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 0.95rem;
    }
    .tt-cell {
        padding: 8px;
        border-radius: 6px;
        color: #1a1a1a;
        font-size: 0.85rem;
        line-height: 1.3;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        min-height: 75px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .tt-subject { font-weight: 700; margin-bottom: 3px; }
    .tt-faculty { font-size: 0.78rem; opacity: 0.85; }
    .tt-room { font-size: 0.75rem; font-weight: 600; margin-top: 3px; }
    .tag-lab { background-color: #fff3bf; color: #f59f00; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .tag-change { background-color: #ffe3e3; color: #e03131; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .tag-ok { background-color: #d3f9d8; color: #2b8a3e; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    </style>
""",
    unsafe_allow_html=True,
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

# Soft palette for subject color coding
SUBJECT_PALETTE = [
    "#D0EBFF", "#E6FCF5", "#FFF3BF", "#FFE3E3", "#F3D9FA",
    "#EEBFA8", "#D3F9D8", "#E0FEFF", "#FFEC99", "#FCC2D7"
]

def get_subject_color(subject: str) -> str:
    """Generates a consistent background hue for a given subject."""
    if not subject or subject == "—":
        return "#ffffff"
    val = sum(ord(c) for c in str(subject).upper())
    return SUBJECT_PALETTE[val % len(SUBJECT_PALETTE)]


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
        return pd.DataFrame(columns=["Faculty_ID", "Faculty_Name", "Mobile", "Is_Coordinator", "Class_Coordinated"])
    id_col = find_col(df, "faculty_id", "id")
    name_col = find_col(df, "faculty_name", "name")
    mobile_col = find_col(df, "mobile", "phone", "contact", "mobile_no")
    coord_col = find_col(df, "is_coordinator", "coordinator", "is_coord")
    class_coord_col = find_col(df, "class_coordinated", "coordinator_class", "class")

    out = pd.DataFrame({
        "Faculty_ID": df[id_col].map(clean) if id_col else "",
        "Faculty_Name": df[name_col].map(clean) if name_col else "",
        "Mobile": df[mobile_col].map(clean) if mobile_col else "",
        "Is_Coordinator": df[coord_col].map(clean) if coord_col else "No",
        "Class_Coordinated": df[class_coord_col].map(clean) if class_coord_col else "",
    })
    out = out[(out["Faculty_ID"] != "") | (out["Faculty_Name"] != "")].copy()
    return out.reset_index(drop=True)


def build_faculty_map(faculty: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    fmap = {}
    for _, r in faculty.iterrows():
        key = norm(r["Faculty_ID"]) or norm(r["Faculty_Name"])
        fmap[key] = {
            "name": clean(r["Faculty_Name"]) or clean(r["Faculty_ID"]),
            "mobile": clean(r["Mobile"]),
        }
    return fmap


def build_coordinator_map(faculty: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    coord_map = {}
    for _, r in faculty.iterrows():
        cls = clean(r["Class_Coordinated"])
        is_coord = norm(r["Is_Coordinator"]) in ["YES", "TRUE", "1"]
        if cls or is_coord:
            target_class = cls if cls else "General"
            coord_map[target_class] = {
                "name": clean(r["Faculty_Name"]),
                "mobile": clean(r["Mobile"]) or "N/A",
            }
    return coord_map


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


def allocate_rooms(timetable: pd.DataFrame, rooms: pd.DataFrame, labs: pd.DataFrame):
    """In-memory deterministic room allocation."""
    if timetable.empty:
        return timetable.copy(), {"Timetable periods": 0}, pd.DataFrame()

    lab_map = build_lab_map(labs)
    all_rooms = [clean(x) for x in rooms["Room_ID"].tolist() if clean(x)]
    room_type = {norm(r["Room_ID"]): clean(r["Type"]) for _, r in rooms.iterrows()}
    theory_rooms = [r for r in all_rooms if room_is_theory_type(room_type.get(norm(r), ""))]
    other_rooms = [r for r in all_rooms if r not in theory_rooms]
    if not theory_rooms:
        theory_rooms = all_rooms[:]
    candidate_base = theory_rooms + [r for r in other_rooms if norm(r) not in {norm(x) for x in theory_rooms}]

    occupancy: Dict[Tuple[str, int], set] = defaultdict(set)
    proposed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    day_order = {d: i for i, d in enumerate(DAYS)}
    ordered = timetable.copy()
    ordered["_d"] = ordered["Day"].map(day_order)
    ordered = ordered.sort_values(["_d", "Period", "Class", "Subject"]).drop(columns=["_d"])

    # Fixed labs allocation
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

    # Theory allocation
    theory = ordered[~ordered["Subject"].map(lambda x: is_lab(x, lab_map))].copy()
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for _, row in theory.iterrows():
        groups[(row["Day"], row["Class"], shift_block(int(row["Period"])))].append(row.to_dict())

    group_items = sorted(groups.items(), key=lambda kv: (-len(kv[1]), day_order[kv[0][0]], kv[0][2], kv[0][1]))
    last_room: Dict[Tuple[str, str], str] = {}
    class_day_rooms: Dict[Tuple[str, str], set] = defaultdict(set)

    for (day, cls, block), group in group_items:
        periods = sorted({int(x["Period"]) for x in group})
        old_rooms = [clean(x.get("Room", "")) for x in group if clean(x.get("Room", ""))]
        old_common = old_rooms[0] if old_rooms and len({norm(x) for x in old_rooms}) == 1 else ""
        candidates = []
        for idx, room in enumerate(candidate_base):
            rk = norm(room)
            if any(rk in {norm(x) for x in occupancy[(day, p)]} for p in periods):
                continue
            score = 0
            if old_common and rk == norm(old_common):
                score += 100000
            prev = last_room.get((cls, day), "")
            if prev and rk == norm(prev):
                score += 50000
            if rk in {norm(x) for x in class_day_rooms[(cls, day)]}:
                score += 15000
            if room in theory_rooms:
                score += 5000
            usage = sum(1 for x in proposed if x["Day"] == day and norm(x["Proposed Room"]) == rk)
            score += min(usage, 20) * 50
            score -= idx
            candidates.append((score, room))

        if not candidates:
            for x in group:
                proposed.append({
                    "Day": day, "Period": int(x["Period"]), "Class": cls, "Subject": x["Subject"],
                    "Faculty": x.get("Faculty", ""), "Old Room": x.get("Room", ""), "Proposed Room": "UNALLOCATED",
                    "Allocation": "FAILED", "Shift Block": block, "Status": "FAILED", "Reason": "No room is free for shift block",
                })
            continue

        room = max(candidates, key=lambda z: z[0])[1]
        last_room[(cls, day)] = room
        class_day_rooms[(cls, day)].add(room)
        for x in group:
            p = int(x["Period"])
            occupancy[(day, p)].add(room)
            old = clean(x.get("Room", ""))
            status = "UNCHANGED" if old and norm(old) == norm(room) else "ROOM CHANGE"
            proposed.append({
                "Day": day, "Period": p, "Class": cls, "Subject": x["Subject"], "Faculty": x.get("Faculty", ""),
                "Old Room": old, "Proposed Room": room, "Allocation": "THEORY-AUTO", "Shift Block": block,
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
    
    used_rooms = int(result[result["Proposed Room"] != "UNALLOCATED"]["Proposed Room"].nunique()) if not result.empty else 0
    current_used = int(timetable.loc[timetable["Room"].ne(""), "Room"].nunique())
    
    metrics = {
        "Timetable periods": total,
        "Allocated": allocated,
        "Unallocated": total - allocated,
        "Lab periods fixed": lab_count,
        "Theory periods": theory_count,
        "Room changes from current timetable": changes,
        "Rooms in Supabase": len(all_rooms),
        "Rooms proposed used": used_rooms,
        "Rooms currently used": current_used,
        "Allocation %": round((allocated / total) * 100, 2) if total else 0.0,
    }
    return result, metrics, pd.DataFrame(failures)


def faculty_display_info(value: str, faculty_map: Dict[str, Dict[str, str]]) -> str:
    v = clean(value)
    if not v:
        return "—"
    finfo = faculty_map.get(norm(v))
    if finfo:
        name = finfo["name"]
        mob = f" ({finfo['mobile']})" if finfo['mobile'] else ""
        return f"{name}{mob}"
    return v


def render_html_grid(result: pd.DataFrame, class_name: str, faculty_map: Dict[str, Dict[str, str]]) -> str:
    """Renders a styled HTML grid for class timetables with colored subject cards."""
    g = result[result["Class"] == class_name].copy()
    html = ["<table style='width:100%; border-collapse: separate; border-spacing: 6px;'>"]
    html.append("<thead><tr style='background-color:#e9ecef; text-align:center;'><th>Day</th>")
    for p in PERIODS:
        html.append(f"<th style='padding:8px;'>Period {p}</th>")
    html.append("</tr></thead><tbody>")

    for day in DAYS:
        html.append(f"<tr><td style='font-weight:bold; background-color:#f1f3f5; text-align:center; padding:8px; border-radius:4px;'>{day}</td>")
        for p in PERIODS:
            x = g[(g["Day"] == day) & (g["Period"] == p)]
            if x.empty:
                html.append("<td style='background-color:#ffffff; border:1px dashed #dee2e6; text-align:center; color:#adb5bd; border-radius:4px;'>—</td>")
            else:
                r = x.iloc[0]
                room = clean(r["Proposed Room"])
                subject = clean(r["Subject"]) or "—"
                fac = faculty_display_info(r["Faculty"], faculty_map)
                bg = get_subject_color(subject)
                
                is_lab_type = r["Allocation"] == "LAB-FIXED"
                badge = " <span class='tag-lab'>LAB</span>" if is_lab_type else ""
                room_str = f"R: {room}" if room != "UNALLOCATED" else "<span style='color:red;'>UNALLOCATED</span>"
                
                cell_content = f"""
                <td style='background-color:{bg}; border-radius:6px; padding:6px;'>
                    <div class='tt-subject'>{subject}{badge}</div>
                    <div class='tt-faculty'>👤 {fac}</div>
                    <div class='tt-room'>🏛️ {room_str}</div>
                </td>
                """
                html.append(cell_content)
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)


def vacancy_grid(result: pd.DataFrame, rooms: pd.DataFrame) -> pd.DataFrame:
    all_rooms = sorted([clean(x) for x in rooms["Room_ID"].tolist() if clean(x)], key=lambda x: x.upper())
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
            vacant = [r for r in all_rooms if norm(r) not in used]
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
    return pd.DataFrame(rows).sort_values(["Occupancy %", "Room"], ascending=[False, True]).reset_index(drop=True)


def room_daily_summary(result: pd.DataFrame, rooms: pd.DataFrame) -> pd.DataFrame:
    all_rooms = [clean(x) for x in rooms["Room_ID"].tolist() if clean(x)]
    total_rooms = len(all_rooms)
    rows = []
    for day in DAYS:
        for p in PERIODS:
            used = result[(result["Day"] == day) & (result["Period"] == p) & (result["Proposed Room"] != "UNALLOCATED")]
            occupied = int(used["Proposed Room"].map(norm).nunique())
            vacant = max(total_rooms - occupied, 0)
            rows.append({
                "Day": day,
                "Period": f"P{p}",
                "Occupied rooms": occupied,
                "Vacant rooms": vacant,
                "Slot occupancy %": round((occupied / total_rooms * 100) if total_rooms else 0.0, 2),
            })
    return pd.DataFrame(rows)


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


def to_excel(timetable: pd.DataFrame, proposed: pd.DataFrame, rooms: pd.DataFrame, occupancy: pd.DataFrame, vacancy: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        timetable.to_excel(writer, sheet_name="Current_Timetable", index=False)
        proposed.to_excel(writer, sheet_name="Proposed_Allocation", index=False)
        rooms.to_excel(writer, sheet_name="Rooms_Master", index=False)
        occupancy.to_excel(writer, sheet_name="Room_Occupancy", index=False)
        vacancy.to_excel(writer, sheet_name="Vacant_Rooms_Week", index=False)
    output.seek(0)
    return output.getvalue()


# ==========================================================
# APPLICATION LAYOUT
# ==========================================================
st.title("🏫 Timetable & Automatic Room Allocation")
st.caption("Interactive, Read-Only Scheduling Dashboard")

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Reload Supabase Data", use_container_width=True):
        fetch_table.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("### 📋 Rules Summary")
    st.write("• Shifts allowed only at P1, P3, P5")
    st.write("• Consistent theory room per shift block")
    st.write("• Fixed mappings enforced for Labs")
    st.write("• Read-Only: No Supabase writes")

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
    st.error(f"Failed to load database tables: {exc}")
    st.stop()

if timetable.empty or rooms.empty:
    st.error("Missing required timetable or rooms data in Supabase.")
    st.stop()

try:
    proposed, metrics, failures = allocate_rooms(timetable, rooms, labs)
except Exception as exc:
    st.error(f"Allocation calculation failed: {exc}")
    st.stop()

# ==========================================================
# SUMMARY METRICS & CHARTS
# ==========================================================
overall = overall_occupancy(proposed, rooms)
cols = st.columns(7)
cols[0].metric("Total Periods", metrics["Timetable periods"])
cols[1].metric("Allocated", metrics["Allocated"])
cols[2].metric("Unallocated", metrics["Unallocated"])
cols[3].metric("Rooms Available", metrics["Rooms in Supabase"])
cols[4].metric("Room Changes", metrics["Room changes from current timetable"])
cols[5].metric("Occupancy %", f"{overall['Overall occupancy %']}%")
cols[6].metric("Vacancy %", f"{overall['Overall vacancy %']}%")

if failures is not None and not failures.empty:
    with st.expander("⚠️ Room Allocation Warnings / Failures"):
        st.dataframe(failures, use_container_width=True, hide_index=True)

st.markdown("---")

classes = sorted(timetable["Class"].unique().tolist())

# ==========================================================
# 1. STANDARD CLASS TIMETABLE (INTERACTIVE VIEW)
# ==========================================================
st.header("1. Interactive Class Timetable")
selected_class = st.selectbox("Select Class/Section", classes, key="main_class_select")

st.markdown(f"#### Class Timetable: **{selected_class}**")
html_grid = render_html_grid(proposed, selected_class, faculty_map)
st.markdown(html_grid, unsafe_allow_html=True)

# ==========================================================
# 2. CURRENT VS PROPOSED ALLOCATION DETAILS
# ==========================================================
st.markdown("---")
st.header("2. Current vs. Proposed Allocations")
selected_rows = proposed[proposed["Class"] == selected_class].copy()
selected_rows["Faculty Info"] = selected_rows["Faculty"].map(lambda f: faculty_display_info(f, faculty_map))

show_cols = ["Day", "Period", "Subject", "Faculty Info", "Old Room", "Proposed Room", "Status", "Allocation"]
st.dataframe(selected_rows[show_cols], use_container_width=True, hide_index=True)

# ==========================================================
# 3. VACANT ROOMS MATRIX
# ==========================================================
st.markdown("---")
st.header("3. Weekly Vacant Rooms Matrix")
vacancy = vacancy_grid(proposed, rooms)
st.dataframe(vacancy, use_container_width=True, hide_index=True)

# ==========================================================
# 4. ROOM OCCUPANCY % & VISUALIZATIONS
# ==========================================================
st.markdown("---")
st.header("4. Room Utilization & Analytics")
occupancy = room_occupancy_summary(proposed, rooms)

col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("Occupancy by Room")
    st.dataframe(occupancy, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Top Utilized Rooms")
    fig = px.bar(
        occupancy.head(10),
        x="Room",
        y="Occupancy %",
        color="Type",
        title="Top 10 Most Utilized Rooms",
        text="Occupancy %",
    )
    fig.update_layout(xaxis_title="Room", yaxis_title="Occupancy (%)", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 5. DAILY UTILIZATION
# ==========================================================
st.markdown("---")
st.header("5. Daily Room Utilization Breakdown")
daily = room_daily_summary(proposed, rooms)

fig_daily = px.line(
    daily,
    x="Period",
    y="Slot occupancy %",
    color="Day",
    markers=True,
    title="Room Occupancy Rate Across Periods (Daily)",
)
st.plotly_chart(fig_daily, use_container_width=True)

# ==========================================================
# 6. ALL CLASS TIMETABLES & COORDINATOR DETAILS
# ==========================================================
st.markdown("---")
st.header("6. All Class Timetables & Class Coordinators")
st.caption("Expand any section below to review schedule details and coordinator contact info.")

search_query = st.text_input("🔍 Search Class or Coordinator Name", "")

for cls in classes:
    # Class Coordinator Info Lookup
    coord_info = coordinator_map.get(cls, {"name": "Not Assigned", "mobile": "N/A"})
    
    # Filter search
    if search_query:
        match_class = search_query.lower() in cls.lower()
        match_coord = search_query.lower() in coord_info["name"].lower()
        if not (match_class or match_coord):
            continue

    with st.expander(f"📘 Class: {cls}  |  Coordinator: {coord_info['name']} (📱 {coord_info['mobile']})"):
        st.markdown(
            f"""
            <div class="coordinator-badge">
                <b>👤 Class Coordinator:</b> {coord_info['name']} &nbsp;|&nbsp; 
                <b>📱 Contact:</b> {coord_info['mobile']}
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Render color-coded HTML timetable grid for the class
        cls_html = render_html_grid(proposed, cls, faculty_map)
        st.markdown(cls_html, unsafe_allow_html=True)

# ==========================================================
# 7. EXPORT & DOWNLOADS
# ==========================================================
st.markdown("---")
st.header("7. Export Reports")
d_col1, d_col2 = st.columns(2)

with d_col1:
    st.download_button(
        "⬇️ Download Proposed Allocation CSV",
        proposed.to_csv(index=False).encode("utf-8"),
        file_name="proposed_room_allocation.csv",
        mime="text/csv",
        use_container_width=True,
    )

with d_col2:
    st.download_button(
        "⬇️ Download Excel Master Report",
        to_excel(timetable, proposed, rooms, occupancy, vacancy),
        file_name="timetable_room_allocation_complete.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.info("ℹ️ **READ-ONLY APP**: This application performs read operations only on Supabase tables and does not write or modify records.")
