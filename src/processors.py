# src/processors.py
import pandas as pd
import os, re
from io import BytesIO


# Helpers

def normalize_name(name: str) -> str:
    return name.strip().lower()

EVENT_KEYWORDS = [
    'event','workshop','hackathon','audition','rehearsal','match','tournament',
    'seminar','competition','tryouts','practice','session','register',
    'meet','meetup','talk','webinar','presentation','contest'
]


# Survey parsing

def parse_survey(file_like):
    df = pd.read_csv(file_like)
    df.columns = [c.strip().lower() for c in df.columns]

    if 'club_name' not in df.columns:
        raise ValueError("Survey CSV must have a 'club_name' column")

    df['club_name'] = df['club_name'].apply(normalize_name)

    # Normalize optional columns
    if 'heard_often' not in df.columns:
        df['heard_often'] = 0
    if 'participation_count' not in df.columns:
        df['participation_count'] = 0
    if 'feedback_text' not in df.columns:
        df['feedback_text'] = ""

    agg = df.groupby('club_name').agg(
        heard_often_mean = ('heard_often', lambda s: pd.to_numeric(s, errors='coerce').mean()),
        participation_mean = ('participation_count', lambda s: pd.to_numeric(s, errors='coerce').mean()),
        num_responses = ('club_name','count')
    ).reset_index()

    return agg


# WhatsApp parsing

def parse_whatsapp_text_file_bytes(content_bytes):
    text = content_bytes.decode('utf-8', errors='ignore')
    lines = [L.strip() for L in text.splitlines() if L.strip()]
    total_msgs = 0
    senders = set()
    sender_pattern = re.compile(r'-\s*(.+?):')

    for L in lines:
        if '-' in L and ':' in L:
            total_msgs += 1
            m = sender_pattern.search(L)
            if m:
                senders.add(m.group(1).strip())
    return {'total_msgs': total_msgs, 'unique_senders': len(senders)}

def parse_whatsapp_folder(uploaded_files):
    rows = []
    for f in uploaded_files:
        fname = getattr(f, "name", None) or os.path.basename(f)
        club_name = normalize_name(os.path.splitext(fname)[0])
        try:
            if hasattr(f, "read"):
                content_bytes = f.read()
            else:
                with open(f, 'rb') as fh:
                    content_bytes = fh.read()
            res = parse_whatsapp_text_file_bytes(content_bytes)
        except Exception:
            res = {'total_msgs': 0, 'unique_senders': 0}
        rows.append({
            'club_name': club_name,
            'whatsapp_msgs': res['total_msgs'],
            'whatsapp_unique_senders': res['unique_senders']
        })
    return pd.DataFrame(rows)


# Event log parsing

def parse_event_log(file_like):
    try:
        df = pd.read_csv(file_like)
    except Exception:
        df = pd.read_excel(file_like)

    df.columns = [c.strip().lower() for c in df.columns]

    if 'club_name' not in df.columns:
        raise ValueError("Event log must have a 'club_name' column")
    if 'event_title' not in df.columns:
        df['event_title'] = ""
    if 'event_description' not in df.columns:
        df['event_description'] = ""

    df['club_name'] = df['club_name'].apply(normalize_name)

    agg = df.groupby('club_name').agg(
        event_count=('event_title','count')
    ).reset_index()

    return df, agg
