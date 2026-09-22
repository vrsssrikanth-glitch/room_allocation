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
# This program is completely separate from tt_supabase.py.
# It reads Supabase timetable/rooms/labs/faculty data only.
# It NEVER writes, updates, inserts or deletes any Supabase data.

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


def allocate_rooms(timetable: pd.DataFrame, rooms: pd.DataFrame, labs: pd.DataFrame):
    """In-memory deterministic room allocation. NO database writes."""
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

    # Fixed labs first
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

    # Theory is allocated one room per class/day/shift block.
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
                    "Allocation": "FAILED", "Shift Block": block, "Status": "FAILED", "Reason": "No room is free for the shift block",
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
    room_type_lookup = {norm(r["Room_ID"]): clean(r["Type"]) for _, r in rooms.iterrows()}
    fallback_theory = 0
    for _, x in result[result["Allocation"] == "THEORY-AUTO"].iterrows():
        if room_is_lab_type(room_type_lookup.get(norm(x["Proposed Room"]), "")):
            fallback_theory += 1

    used_rooms = int(result[result["Proposed Room"] != "UNALLOCATED"]["Proposed Room"].nunique()) if not result.empty else 0
    current_used = int(timetable.loc[timetable["Room"].ne(""), "Room"].nunique())
    metrics = {
        "Timetable periods": total,
        "Allocated": allocated,
        "Unallocated": total - allocated,
        "Lab periods fixed": lab_count,
        "Theory periods": theory_count,
        "Theory using lab-type rooms": fallback_theory,
        "Room changes from current timetable": changes,
        "Rooms in Supabase": len(all_rooms),
        "Rooms proposed used": used_rooms,
        "Rooms currently used": current_used,
        "Allocation %": round((allocated / total) * 100, 2) if total else 0.0,
    }
    return result, metrics, pd.DataFrame(failures)


def faculty_display(value: str, faculty_map: Dict[str, str]) -> str:
    v = clean(value)
    if not v:
        return "Faculty: —"
    return f"Faculty: {faculty_map.get(norm(v), v)}"


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
                room_line = f"Room: {room} [LAB]"
            elif room == "UNALLOCATED":
                room_line = "Room: UNALLOCATED"
            else:
                room_line = f"Room: {room}"
            row[f"P{p}"] = f"{subject}\n{faculty}\n{room_line}"
        rows.append(row)
    return pd.DataFrame(rows)


def all_class_timetables(result: pd.DataFrame, faculty_map: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    return {cls: class_timetable_grid(result, cls, faculty_map) for cls in sorted(result["Class"].unique())}


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
                "Period": p,
                "Occupied rooms": occupied,
                "Vacant rooms": vacant,
                "Slot occupancy %": round((occupied / total_rooms * 100) if total_rooms else 0.0, 2),
            })
    return pd.DataFrame(rows)


def to_excel(timetable: pd.DataFrame, proposed: pd.DataFrame, rooms: pd.DataFrame, occupancy: pd.DataFrame, vacancy: pd.DataFrame, class_grids: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        timetable.to_excel(writer, sheet_name="Current_Timetable", index=False)
        proposed.to_excel(writer, sheet_name="Proposed_Allocation", index=False)
        rooms.to_excel(writer, sheet_name="Rooms_Master", index=False)
        occupancy.to_excel(writer, sheet_name="Room_Occupancy", index=False)
        vacancy.to_excel(writer, sheet_name="Vacant_Rooms_Week", index=False)
        for cls, grid in class_grids.items():
            safe = "CLASS_" + "".join(ch if ch not in '\\/:*?[]' else '_' for ch in str(cls))
            safe = safe[:31]
            grid.to_excel(writer, sheet_name=safe, index=False)
    output.seek(0)
    return output.getvalue()


# ==========================================================
# APP
# ==========================================================
st.title("🏫 Standard Timetable + Automatic Room Allocation")
st.caption("Independent read-only application. It does not modify tt_supabase.py or write back to Supabase.")

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reload Supabase data", use_container_width=True):
        fetch_table.clear()
        st.rerun()
    st.markdown("### Allocation rules")
    st.write("• Theory room changes only at P1, P3 and P5")
    st.write("• One room per class within P1–P2, P3–P4 and P5–P7")
    st.write("• Lab subjects use the fixed room from labs table")
    st.write("• Lab/computer/workshop rooms can be used before/after a lab when free")
    st.write("• All rooms in the rooms table are considered")
    st.write("• Read-only: no database writes")

try:
    with st.spinner("Reading timetable, rooms, labs and faculty from Supabase..."):
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
    st.info("Check SUPABASE_URL, SUPABASE_KEY, and table/column names.")
    st.stop()

if timetable.empty:
    st.warning("No timetable rows were found in the Supabase timetable table.")
    st.stop()
if rooms.empty:
    st.error("No rooms are available in the rooms table.")
    st.stop()

try:
    proposed, metrics, failures = allocate_rooms(timetable, rooms, labs)
except Exception as exc:
    st.error(f"Room allocation failed: {exc}")
    st.stop()

# ==========================================================
# KPI / OCCUPANCY
# ==========================================================
overall = overall_occupancy(proposed, rooms)
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Timetable periods", metrics["Timetable periods"])
m2.metric("Allocated", metrics["Allocated"])
m3.metric("Unallocated", metrics["Unallocated"])
m4.metric("Rooms", metrics["Rooms in Supabase"])
m5.metric("Room changes", metrics["Room changes from current timetable"])
m6.metric("Overall occupancy", f"{overall['Overall occupancy %']}%")
m7.metric("Overall vacancy", f"{overall['Overall vacancy %']}%")

if metrics["Unallocated"] == 0:
    st.success("All timetable periods have a proposed room assignment.")
else:
    st.warning(f"{metrics['Unallocated']} timetable periods could not be assigned a room.")

if failures is not None and not failures.empty:
    with st.expander("⚠️ Lab / fixed-room problems"):
        st.dataframe(failures, use_container_width=True, hide_index=True)

classes = sorted(timetable["Class"].unique().tolist())
selected_class = st.selectbox("Select class for standard timetable view", classes)

# ==========================================================
# STANDARD CLASS TIMETABLE
# ==========================================================
st.header(f"1. Standard Timetable – {selected_class}")
st.caption("Each cell shows Subject, Faculty Name and Room Number. Room changes are allowed only at P1, P3 and P5.")
selected_grid = class_timetable_grid(proposed, selected_class, faculty_map)
st.dataframe(selected_grid, use_container_width=True, hide_index=True, height=390)

# ==========================================================
# CURRENT VS PROPOSED DETAILS
# ==========================================================
st.subheader("2. Current vs Proposed Room Allocation")
selected_rows = proposed[proposed["Class"] == selected_class].copy()
show_cols = ["Day", "Period", "Subject", "Faculty", "Old Room", "Proposed Room", "Status", "Allocation", "Shift Block"]
st.dataframe(selected_rows[show_cols], use_container_width=True, hide_index=True, height=420)

# ==========================================================
# VACANT ROOMS THROUGHOUT WEEK
# ==========================================================
st.header("3. Vacant Rooms Throughout the Week")
st.caption("Each cell lists all rooms that are not occupied in that day/period under the proposed allocation.")
vacancy = vacancy_grid(proposed, rooms)
st.dataframe(vacancy, use_container_width=True, hide_index=True, height=390)

# ==========================================================
# ROOM OCCUPANCY SUMMARY
# ==========================================================
st.header("4. Room Occupancy %")
st.caption(f"Weekly occupancy is calculated over {len(DAYS)} days × {len(PERIODS)} periods = {len(DAYS) * len(PERIODS)} room-period slots per room.")
occupancy = room_occupancy_summary(proposed, rooms)
st.dataframe(occupancy, use_container_width=True, hide_index=True, height=520)

# Overall / daily vacancy summary
st.subheader("5. Daily Room Utilization")
daily = room_daily_summary(proposed, rooms)
st.dataframe(daily, use_container_width=True, hide_index=True, height=420)

# ==========================================================
# ALL CLASS TIMETABLES
# ==========================================================
st.header("6. All Class Timetables")
st.caption("Use the expanders to view the standard timetable for every class.")
class_grids = all_class_timetables(proposed, faculty_map)
for cls, grid in class_grids.items():
    with st.expander(f"📘 {cls}"):
        st.dataframe(grid, use_container_width=True, hide_index=True, height=315)

# ==========================================================
# EXPORT
# ==========================================================
st.header("7. Export Reports")
left, right = st.columns(2)
with left:
    st.download_button(
        "⬇️ Download Proposed Allocation CSV",
        proposed.to_csv(index=False).encode("utf-8"),
        file_name="proposed_room_allocation.csv",
        mime="text/csv",
        use_container_width=True,
    )
with right:
    st.download_button(
        "⬇️ Download Complete Excel Report",
        to_excel(timetable, proposed, rooms, occupancy, vacancy, class_grids),
        file_name="timetable_room_allocation_complete.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.info(
    "READ-ONLY: This independent application only reads timetable, rooms, labs and faculty data from Supabase. "
    "It never updates or synchronizes the original timetable application."
)
