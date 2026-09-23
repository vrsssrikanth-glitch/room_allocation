import io
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from supabase import Client, create_client

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Timetable & Automatic Room Allocation",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
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

# Dark vibrant text colors for subject text
TEXT_COLORS = [
    "#1c7ed6", "#2b8a3e", "#d9480f", "#a61e4d", "#5f3dc4",
    "#0c8599", "#e67700", "#c2255c", "#364fc7", "#2f9e44"
]

def get_subject_color(subject: str) -> str:
    """Returns a vibrant text color based on subject name hash."""
    if not subject or subject == "—":
        return "#495057"
    val = sum(ord(c) for c in str(subject).upper())
    return TEXT_COLORS[val % len(TEXT_COLORS)]


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
    faculty_col = find_col(df, "faculty_id", "faculty", "faculty_name", "code")
    day_col = find_col(df, "day")
    period_col = find_col(df, "period")
    room_col = find_col(df, "room", "room_id")

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
    out = pd.DataFrame({"Lab_Subject": df[subject_col].map(clean), "Room": df[room_col].map(clean)})
    return out.reset_index(drop=True)


def normalize_faculty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Faculty_ID", "Faculty_Name", "Mobile", "Is_Coordinator", "Class_Coordinated"])
    id_col = find_col(df, "code", "faculty_id", "id")
    name_col = find_col(df, "name", "faculty_name")
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
    return out.reset_index(drop=True)


def build_faculty_map(faculty: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    fmap = {}
    for _, r in faculty.iterrows():
        key_id = norm(r["Faculty_ID"])
        key_name = norm(r["Faculty_Name"])
        val = {
            "name": clean(r["Faculty_Name"]) or clean(r["Faculty_ID"]),
            "mobile": clean(r["Mobile"]),
        }
        if key_id:
            fmap[key_id] = val
        if key_name:
            fmap[key_name] = val
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


def shift_block(period: int) -> str:
    return SHIFT_BLOCKS[int(period)]


def allocate_rooms(timetable: pd.DataFrame, rooms: pd.DataFrame, labs: pd.DataFrame):
    if timetable.empty:
        return timetable.copy(), {"Timetable periods": 0}, pd.DataFrame()

    lab_map = build_lab_map(labs)
    all_rooms = [clean(x) for x in rooms["Room_ID"].tolist() if clean(x)]
    
    occupancy: Dict[Tuple[str, int], set] = defaultdict(set)
    proposed: List[Dict[str, Any]] = []

    day_order = {d: i for i, d in enumerate(DAYS)}
    ordered = timetable.copy()
    ordered["_d"] = ordered["Day"].map(day_order)
    ordered = ordered.sort_values(["_d", "Period", "Class", "Subject"]).drop(columns=["_d"])

    for _, row in ordered.iterrows():
        p = int(row["Period"])
        day = row["Day"]
        cls = row["Class"]
        subj = row["Subject"]
        fac = row["Faculty"]
        
        assigned_room = row["Room"] if row["Room"] else (all_rooms[0] if all_rooms else "A101")
        
        proposed.append({
            "Day": day, "Period": p, "Class": cls, "Subject": subj,
            "Faculty": fac, "Old Room": row["Room"], "Proposed Room": assigned_room,
            "Allocation": "LAB-FIXED" if is_lab(subj, lab_map) else "THEORY-AUTO",
            "Shift Block": shift_block(p), "Status": "ALLOCATED",
        })

    result = pd.DataFrame(proposed)
    metrics = {"Timetable periods": len(timetable), "Allocated": len(result)}
    return result, metrics, pd.DataFrame()


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


def render_native_grid(result: pd.DataFrame, class_name: str, faculty_map: Dict[str, Dict[str, str]]):
    """Renders a grid with colored fonts using Pandas Styler in Streamlit."""
    g = result[result["Class"] == class_name].copy()
    
    grid_data = {f"P{p}": [] for p in PERIODS}
    color_map = {}

    for day_idx, day in enumerate(DAYS):
        for p in PERIODS:
            col_name = f"P{p}"
            x = g[(g["Day"] == day) & (g["Period"] == p)]
            if x.empty:
                grid_data[col_name].append("—")
                color_map[(day_idx, col_name)] = "#adb5bd"
            else:
                r = x.iloc[0]
                subj = clean(r["Subject"]) or "—"
                fac = faculty_display_info(r["Faculty"], faculty_map)
                room = clean(r["Proposed Room"])
                
                text_content = f"{subj}\n👤 {fac}\n🏛️ R: {room}"
                grid_data[col_name].append(text_content)
                color_map[(day_idx, col_name)] = get_subject_color(subj)

    df_grid = pd.DataFrame(grid_data, index=DAYS)

    def style_cells(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for (r_idx, col), color in color_map.items():
            styles.iloc[r_idx, df.columns.get_loc(col)] = f"color: {color}; font-weight: bold; white-space: pre-line;"
        return styles

    styled_df = df_grid.style.apply(style_cells, axis=None)
    st.dataframe(styled_df, use_container_width=True, height=255)


# ==========================================================
# APPLICATION INTERFACE
# ==========================================================
st.title("🏫 Timetable & Automatic Room Allocation")

with st.sidebar:
    if st.button("🔄 Reload Supabase Data", use_container_width=True):
        fetch_table.clear()
        st.rerun()

try:
    timetable = normalize_timetable(fetch_table(TABLE_TIMETABLE))
    rooms = normalize_rooms(fetch_table(TABLE_ROOMS))
    labs = normalize_labs(fetch_table(TABLE_LABS))
    faculty = normalize_faculty(fetch_table(TABLE_FACULTY))
    
    faculty_map = build_faculty_map(faculty)
    coordinator_map = build_coordinator_map(faculty)
    proposed, metrics, failures = allocate_rooms(timetable, rooms, labs)
except Exception as exc:
    st.error(f"Error loading data: {exc}")
    st.stop()

classes = sorted(timetable["Class"].unique().tolist())

# ----------------------------------------------------------
# 1. CLASS TIMETABLE (COLORED FONT GRID)
# ----------------------------------------------------------
st.header("1. Interactive Class Timetable")
selected_class = st.selectbox("Select Class/Section", classes, key="main_class_select")

# Class Coordinator Display
coord_info = coordinator_map.get(selected_class, {"name": "Not Assigned", "mobile": "N/A"})
st.info(f"👤 **Class Coordinator:** {coord_info['name']} &nbsp;|&nbsp; 📱 **Mobile:** {coord_info['mobile']}")

# Colored Font Timetable Grid
render_native_grid(proposed, selected_class, faculty_map)

