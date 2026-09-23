import io
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from supabase import Client, create_client

# ==========================================================
# SEPARATE READ-ONLY ROOM ALLOCATION APPLICATION
# ==========================================================
# IMPORTANT:
# - This program is independent of tt_supabase.py.
# - It reads timetable/rooms/labs directly from Supabase.
# - It NEVER inserts, updates, deletes, or syncs any data.
# - Automatic room allocation exists only in memory in this app.

st.set_page_config(
    page_title="Automatic Room Allocation – Timetable",
    page_icon="🏫",
    layout="wide",
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


def clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def norm(value: Any) -> str:
    return clean(value).upper()


def env_value(name: str) -> str:
    # Supports both Streamlit Cloud secrets and local environment variables.
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
    timetable = fetch_table(TABLE_TIMETABLE)
    rooms = fetch_table(TABLE_ROOMS)
    labs = fetch_table(TABLE_LABS)
    return timetable, rooms, labs


def normalize_timetable(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Class", "Subject", "Faculty", "Day", "Period", "Room"])

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


def build_lab_map(labs: pd.DataFrame) -> Dict[str, str]:
    return {norm(r["Lab_Subject"]): clean(r["Room"]) for _, r in labs.iterrows()}


def shift_block(period: int) -> str:
    return SHIFT_BLOCKS[int(period)]


def slot_key(day: str, period: int) -> Tuple[str, int]:
    return day, int(period)


def allocate_rooms(
    timetable: pd.DataFrame,
    rooms: pd.DataFrame,
    labs: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Greedy, deterministic, read-only optimizer.

    Rules implemented:
      - Laboratory subjects use the fixed room from labs table.
      - Theory room changes can happen only at P1, P3, and P5.
      - A theory class uses one room throughout a shift block:
        P1-P2, P3-P4, or P5-P7.
      - Any room in the rooms table is a candidate.
      - Lab/computer/workshop rooms may be used by theory when not occupied by a lab,
        allowing before/after-lab use.
      - Existing room is preserved when possible.
      - Previous block room is strongly preferred to reduce movement.
      - No database writes are performed.
    """
    if timetable.empty:
        return timetable.copy(), {"total": 0}, pd.DataFrame()

    lab_map = build_lab_map(labs)

    room_records = rooms.to_dict("records")
    all_rooms = [clean(r["Room_ID"]) for r in room_records if clean(r["Room_ID"])]
    room_type = {norm(r["Room_ID"]): clean(r["Type"]) for r in room_records}

    theory_rooms = [r for r in all_rooms if room_is_theory_type(room_type.get(norm(r), ""))]
    other_rooms = [r for r in all_rooms if r not in theory_rooms]
    if not theory_rooms:
        theory_rooms = all_rooms[:]

    # Occupancy of proposed allocation: (day, period) -> set(room)
    occupancy: Dict[Tuple[str, int], set] = defaultdict(set)
    proposed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    ordered = timetable.sort_values(
        by=["Day", "Period", "Class", "Subject"],
        key=lambda s: s.map(DAYS.index) if s.name == "Day" else s,
    ).copy()

    # Allocate fixed laboratories first.
    for _, row in ordered.iterrows():
        if not is_lab(row["Subject"], lab_map):
            continue

        mapped = lab_map.get(norm(row["Subject"]))
        if not mapped:
            failures.append(
                {
                    "Day": row["Day"],
                    "Period": row["Period"],
                    "Class": row["Class"],
                    "Subject": row["Subject"],
                    "Reason": "Lab subject detected, but no mapping exists in labs table",
                }
            )
            continue

        if norm(mapped) not in {norm(x) for x in all_rooms}:
            failures.append(
                {
                    "Day": row["Day"],
                    "Period": row["Period"],
                    "Class": row["Class"],
                    "Subject": row["Subject"],
                    "Reason": f"Lab room {mapped} is not present in rooms table",
                }
            )
            continue

        key = slot_key(row["Day"], row["Period"])
        if norm(mapped) in {norm(x) for x in occupancy[key]}:
            failures.append(
                {
                    "Day": row["Day"],
                    "Period": row["Period"],
                    "Class": row["Class"],
                    "Subject": row["Subject"],
                    "Reason": f"Fixed lab room clash: {mapped}",
                }
            )
            continue

        occupancy[key].add(mapped)
        proposed.append(
            {
                "Day": row["Day"],
                "Period": int(row["Period"]),
                "Class": row["Class"],
                "Subject": row["Subject"],
                "Faculty": row["Faculty"],
                "Old Room": row["Room"],
                "Proposed Room": mapped,
                "Allocation": "LAB-FIXED",
                "Shift Block": "LAB",
                "Status": "LAB FIXED",
                "Reason": "",
            }
        )

    # Theory groups: class + day + permitted change block.
    theory_rows = ordered[~ordered["Subject"].map(lambda x: is_lab(x, lab_map))].copy()
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for _, row in theory_rows.iterrows():
        key = (row["Day"], row["Class"], shift_block(int(row["Period"])))
        groups[key].append(row.to_dict())

    # Prioritize hard blocks first: more periods, earlier in the week, then class.
    group_items = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), DAYS.index(kv[0][0]), kv[0][2], kv[0][1]),
    )

    last_room_for_class_day: Dict[Tuple[str, str], str] = {}
    same_class_day_rooms: Dict[Tuple[str, str], set] = defaultdict(set)

    candidate_base = theory_rooms + [r for r in other_rooms if norm(r) not in {norm(x) for x in theory_rooms}]

    for (day, cls, block), group in group_items:
        periods = sorted({int(x["Period"]) for x in group})
        old_rooms = [clean(x.get("Room", "")) for x in group if clean(x.get("Room", ""))]
        old_common = ""
        if old_rooms and len({norm(x) for x in old_rooms}) == 1:
            old_common = old_rooms[0]

        candidates: List[Tuple[int, str]] = []
        occupied_by_slot = {p: {norm(x) for x in occupancy[slot_key(day, p)]} for p in periods}

        for idx, room in enumerate(candidate_base):
            rkey = norm(room)
            if any(rkey in occupied_by_slot[p] for p in periods):
                continue

            score = 0
            # Strong preference: preserve the currently stored room.
            if old_common and rkey == norm(old_common):
                score += 100000
            # Strong preference: keep same room across the allowed shift boundary.
            prev = last_room_for_class_day.get((cls, day), "")
            if prev and rkey == norm(prev):
                score += 50000
            # Reuse one of this class's already used rooms on the same day.
            if any(rkey == norm(x) for x in same_class_day_rooms[(cls, day)]):
                score += 15000
            # Prefer a normal theory room; lab-type rooms are fallback capacity.
            if room in theory_rooms:
                score += 5000
            # Prefer rooms already used across the day to reduce total rooms used.
            # This is only a soft objective; collision-free allocation has priority.
            usage_hint = sum(
                1
                for x in proposed
                if x["Day"] == day and norm(x["Proposed Room"]) == rkey
            )
            score += min(usage_hint, 20) * 50
            score -= idx
            candidates.append((score, room))

        if not candidates:
            for x in group:
                proposed.append(
                    {
                        "Day": day,
                        "Period": int(x["Period"]),
                        "Class": cls,
                        "Subject": x["Subject"],
                        "Faculty": x.get("Faculty", ""),
                        "Old Room": x.get("Room", ""),
                        "Proposed Room": "UNALLOCATED",
                        "Allocation": "FAILED",
                        "Shift Block": block,
                        "Status": "FAILED",
                        "Reason": "No room is free for all periods in this shift block",
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
                    "Allocation": "THEORY-AUTO",
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
                "Proposed Room", "Allocation", "Shift Block", "Status", "Reason"
            ]
        )

    day_order = {d: i for i, d in enumerate(DAYS)}
    result["_day_order"] = result["Day"].map(day_order)
    result = result.sort_values(["_day_order", "Class", "Period", "Subject"]).drop(columns=["_day_order"])
    result = result.reset_index(drop=True)

    total = len(timetable)
    allocated_count = int((result["Proposed Room"] != "UNALLOCATED").sum()) if not result.empty else 0
    lab_count = int((result["Allocation"] == "LAB-FIXED").sum()) if not result.empty else 0
    theory_count = int((result["Allocation"] == "THEORY-AUTO").sum()) if not result.empty else 0
    changes = int((result["Status"] == "ROOM CHANGE").sum()) if not result.empty else 0
    fallback_theory = 0
    room_type_lookup = {norm(k): room_type.get(norm(k), "") for k in all_rooms}
    if not result.empty:
        for _, r in result[result["Allocation"] == "THEORY-AUTO"].iterrows():
            if room_is_lab_type(room_type_lookup.get(norm(r["Proposed Room"]), "")):
                fallback_theory += 1

    room_usage = result[result["Proposed Room"] != "UNALLOCATED"]["Proposed Room"].nunique() if not result.empty else 0
    unique_current = timetable["Room"].replace("", pd.NA).dropna().nunique()

    # Count actual block transitions per class/day, not period changes within a block.
    block_changes = 0
    class_day_block_rooms: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for _, r in result[result["Allocation"] == "THEORY-AUTO"].iterrows():
        class_day_block_rooms[(r["Class"], r["Day"])][r["Shift Block"]] = r["Proposed Room"]
    block_order = {"P1-P2": 1, "P3-P4": 2, "P5-P7": 3}
    for key, mapping in class_day_block_rooms.items():
        blocks = sorted(mapping, key=lambda b: block_order[b])
        for a, b in zip(blocks, blocks[1:]):
            if norm(mapping[a]) != norm(mapping[b]):
                block_changes += 1

    metrics = {
        "Timetable periods": total,
        "Allocated": allocated_count,
        "Unallocated": total - allocated_count,
        "Lab periods fixed": lab_count,
        "Theory periods": theory_count,
        "Theory using lab-type rooms": fallback_theory,
        "Room changes from current timetable": changes,
        "Allowed block-to-block room changes": block_changes,
        "Rooms in Supabase": len(all_rooms),
        "Rooms proposed used": int(room_usage),
        "Rooms currently used": int(unique_current),
        "Allocation %": round((allocated_count / total) * 100, 2) if total else 0.0,
    }

    return result, metrics, pd.DataFrame(failures)


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
                subject = x["Subject"]
                if x["Allocation"] == "LAB-FIXED":
                    row[f"P{p}"] = f"{room} [LAB]"
                elif room == "UNALLOCATED":
                    row[f"P{p}"] = "UNALLOCATED"
                else:
                    row[f"P{p}"] = room
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_day"] = out["Day"].map({d: i for i, d in enumerate(DAYS)})
    return out.sort_values(["_day", "Class"]).drop(columns=["_day"]).reset_index(drop=True)


def room_occupancy(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return pd.DataFrame()
    rows = []
    for room, g in result[result["Proposed Room"] != "UNALLOCATED"].groupby("Proposed Room"):
        row = {"Room": room}
        for d in DAYS:
            for p in PERIODS:
                x = g[(g["Day"] == d) & (g["Period"] == p)]
                cell = ""
                if not x.empty:
                    z = x.iloc[0]
                    cell = f"{z['Class']} / {z['Subject']}"
                    if z["Allocation"] == "LAB-FIXED":
                        cell += " [LAB]"
                row[f"{d[:3]} P{p}"] = cell
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Room").reset_index(drop=True)


def timetable_grid(timetable: pd.DataFrame, cls: str | None = None, day: str | None = None) -> pd.DataFrame:
    df = timetable.copy()
    if cls and cls != "All":
        df = df[df["Class"] == cls]
    if day and day != "All":
        df = df[df["Day"] == day]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for d in DAYS if day in (None, "All") else [day]:
        for c in sorted(df["Class"].unique()):
            g = df[(df["Day"] == d) & (df["Class"] == c)]
            if g.empty:
                continue
            row = {"Day": d, "Class": c}
            for p in PERIODS:
                x = g[g["Period"] == p]
                if x.empty:
                    row[f"P{p}"] = "—"
                else:
                    z = x.iloc[0]
                    room = z["Room"] or "(no room)"
                    row[f"P{p}"] = f"{z['Subject']}\n[{room}]"
            rows.append(row)
    return pd.DataFrame(rows)


def to_excel(timetable: pd.DataFrame, proposed: pd.DataFrame, mapping: pd.DataFrame, rooms: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        timetable.to_excel(writer, sheet_name="Current_Timetable", index=False)
        proposed.to_excel(writer, sheet_name="Proposed_Allocation", index=False)
        mapping.to_excel(writer, sheet_name="Class_Room_Mapping", index=False)
        rooms.to_excel(writer, sheet_name="Rooms_Master", index=False)
    output.seek(0)
    return output.getvalue()


st.title("🏫 Automatic Room Allocation – Independent Program")
st.caption(
    "READ-ONLY mode: this application reads Supabase data and calculates room assignments in memory. "
    "It does not modify the timetable, rooms, labs, or any other Supabase table."
)

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reload Supabase data", use_container_width=True):
        fetch_table.clear()
        st.rerun()

    st.markdown("### Allocation rules")
    st.write("• Theory room changes only at P1, P3, and P5")
    st.write("• One room per class within P1–P2, P3–P4, P5–P7")
    st.write("• Lab rooms are fixed from the labs table")
    st.write("• Lab/computer/workshop rooms may be reused before/after labs")
    st.write("• All rooms from the rooms table are candidates")
    st.write("• No database writes")

try:
    with st.spinner("Reading timetable, rooms and lab information from Supabase..."):
        raw_timetable, raw_rooms, raw_labs = load_data()
        timetable = normalize_timetable(raw_timetable)
        rooms = normalize_rooms(raw_rooms)
        labs = normalize_labs(raw_labs)
except Exception as exc:
    st.error(f"Could not read Supabase data: {exc}")
    st.info("Check SUPABASE_URL, SUPABASE_KEY, and the table/column names.")
    st.stop()

if timetable.empty:
    st.warning("No timetable rows were found in the Supabase 'timetable' table.")
    st.stop()

try:
    proposed, metrics, failures = allocate_rooms(timetable, rooms, labs)
except Exception as exc:
    st.error(f"Room allocation failed: {exc}")
    st.stop()

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Timetable periods", metrics["Timetable periods"])
m2.metric("Allocated", metrics["Allocated"])
m3.metric("Unallocated", metrics["Unallocated"])
m4.metric("Room changes", metrics["Room changes from current timetable"])
m5.metric("Rooms available", metrics["Rooms in Supabase"])
m6.metric("Allocation %", f"{metrics['Allocation %']}%")

if metrics["Unallocated"] == 0:
    st.success("All timetable periods received a valid proposed room assignment.")
else:
    st.warning(f"{metrics['Unallocated']} timetable periods could not be allocated.")

if failures is not None and not failures.empty:
    with st.expander("⚠️ Lab/fixed-room problems"):
        st.dataframe(failures, use_container_width=True, hide_index=True)

classes = ["All"] + sorted(timetable["Class"].unique().tolist())
view_day_options = ["All"] + DAYS
col1, col2 = st.columns(2)
with col1:
    selected_class = st.selectbox("Class", classes)
with col2:
    selected_day = st.selectbox("Day", view_day_options)

# ----------------------------------------------------------
# Current timetable
# ----------------------------------------------------------
st.subheader("1. Current Timetable (from Supabase)")
current_grid = timetable_grid(timetable, selected_class, selected_day)
if current_grid.empty:
    st.info("No timetable entries for the selected filter.")
else:
    st.dataframe(current_grid, use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# Proposed allocation
# ----------------------------------------------------------
st.subheader("2. Proposed Automatic Room Allocation")
proposed_display = proposed.copy()
if selected_class != "All":
    proposed_display = proposed_display[proposed_display["Class"] == selected_class]
if selected_day != "All":
    proposed_display = proposed_display[proposed_display["Day"] == selected_day]

show_cols = [
    "Day", "Period", "Class", "Subject", "Faculty", "Old Room",
    "Proposed Room", "Allocation", "Shift Block", "Status", "Reason"
]
st.dataframe(
    proposed_display[show_cols],
    use_container_width=True,
    hide_index=True,
    height=520,
)

# ----------------------------------------------------------
# Class -> room mapping
# ----------------------------------------------------------
st.subheader("3. Class → Room Mapping")
mapping = class_day_mapping(proposed)
mapping_display = mapping.copy()
if selected_class != "All":
    mapping_display = mapping_display[mapping_display["Class"] == selected_class]
if selected_day != "All":
    mapping_display = mapping_display[mapping_display["Day"] == selected_day]
st.dataframe(mapping_display, use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# Room occupancy
# ----------------------------------------------------------
st.subheader("4. Proposed Room Occupancy")
occupancy = room_occupancy(proposed)
st.dataframe(occupancy, use_container_width=True, hide_index=True, height=500)

# ----------------------------------------------------------
# Details
# ----------------------------------------------------------
st.subheader("5. Allocation Details")
detail_cols = st.columns(4)
detail_cols[0].write(f"**Lab periods fixed:** {metrics['Lab periods fixed']}")
detail_cols[1].write(f"**Theory periods:** {metrics['Theory periods']}")
detail_cols[2].write(f"**Theory using lab-type rooms:** {metrics['Theory using lab-type rooms']}")
detail_cols[3].write(f"**Allowed block changes:** {metrics['Allowed block-to-block room changes']}")
detail_cols2 = st.columns(3)
detail_cols2[0].write(f"**Rooms currently used:** {metrics['Rooms currently used']}")
detail_cols2[1].write(f"**Rooms proposed used:** {metrics['Rooms proposed used']}")
detail_cols2[2].write("**Database status:** READ ONLY")

# ----------------------------------------------------------
# Downloads
# ----------------------------------------------------------
st.subheader("6. Export")
left, right = st.columns(2)
with left:
    st.download_button(
        "⬇️ Download Proposed Allocation CSV",
        proposed[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="proposed_room_allocation.csv",
        mime="text/csv",
        use_container_width=True,
    )
with right:
    st.download_button(
        "⬇️ Download Full Excel Report",
        to_excel(timetable, proposed[show_cols], mapping, rooms),
        file_name="automatic_room_allocation_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.info(
    "This program is intentionally separate from tt_supabase.py. "
    "It does not import it, call it, update timetable rows, or write any room allocation back to Supabase. "
    "The proposed allocation disappears when the app session is restarted unless you export the report."
)
