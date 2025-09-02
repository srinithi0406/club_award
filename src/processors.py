# src/processors.py
import pandas as pd
import os, re
import numpy as np
from io import BytesIO
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


# Ensure VADER is available
nltk.download('vader_lexicon', quiet=True)


EVENT_KEYWORDS = [
    'event','workshop','hackathon','audition','rehearsal','match','tournament',
    'seminar','competition','tryouts','practice','session','register','register by',
    'meet','meetup','talk','webinar','presentation','contest'
]


CATEGORY_KEYWORDS = {
    'tech': ['coding', 'robotics', 'programming', 'hackathon', 'python', 'java', 'embedded', 'electronics'],
    'sports': ['football', 'basketball', 'cricket', 'tennis', 'soccer', 'training', 'match', 'tournament', 'camp', 'practice', 'tryouts'],
    'entertainment': ['music', 'dance', 'drama', 'theatre', 'acoustic', 'play', 'singing', 'performance', 'recording'],
    'literature & knowledge': ['quiz', 'debate', 'tamil', 'lecture', 'knowledge', 'mun', 'cultural', 'seminar', 'talk']
}


def normalize_name(name: str) -> str:
    return name.strip().lower()




# Survey parsing


def parse_survey(file_like):
    df = pd.read_csv(file_like)
    df['club_name'] = df['club_name'].apply(normalize_name)


    df.columns = [c.strip() for c in df.columns]
    if 'club_name' not in df.columns:
        raise ValueError("Survey CSV must have a 'club_name' column")


    # normalize optional columns
    if 'heard_often' not in df.columns:
        for alt in ['awareness','how_often','heard']:
            if alt in df.columns:
                df = df.rename(columns={alt:'heard_often'})
                break
    if 'participation_count' not in df.columns and 'participated' in df.columns:
        df['participation_count'] = df['participated'].apply(lambda x: 1 if str(x).strip().lower() in ['yes','y','1','true'] else 0)
    if 'participation_count' not in df.columns:
        df['participation_count'] = 0
    if 'feedback_text' not in df.columns:
        for alt in ['review_text','review','feedback','comments']:
            if alt in df.columns:
                df = df.rename(columns={alt:'feedback_text'})
                break
        if 'feedback_text' not in df.columns:
            df['feedback_text'] = ""


    agg = df.groupby('club_name').agg(
        heard_often_mean = ('heard_often', lambda s: pd.to_numeric(s, errors='coerce').mean()),
        participation_mean = ('participation_count', lambda s: pd.to_numeric(s, errors='coerce').mean()),
        num_responses = ('club_name','count'),
        feedback_concat = ('feedback_text', lambda s: " ".join([str(x) for x in s.dropna() if str(x).strip()!='']))
    ).reset_index()


    agg['heard_often_mean'] = agg['heard_often_mean'].fillna(0)
    agg['participation_mean'] = agg['participation_mean'].fillna(0)
    return agg



# WhatsApp parsing


def parse_whatsapp_text_file_bytes(content_bytes):
    text = content_bytes.decode('utf-8', errors='ignore')
    lines = [L.strip() for L in text.splitlines() if L.strip()]
    total_msgs = 0
    senders = set()
    event_mentions = 0
    sender_pattern = re.compile(r'-\s*(.+?):')
    for L in lines:
        if '-' in L and ':' in L:
            total_msgs += 1
            m = sender_pattern.search(L)
            if m:
                sender = m.group(1).strip()
                senders.add(sender)
            low = L.lower()
            if any(k in low for k in EVENT_KEYWORDS):
                event_mentions += 1
    return {'total_msgs': total_msgs, 'unique_senders': len(senders), 'event_mentions': event_mentions}


def parse_whatsapp_folder(uploaded_files):
    rows = []
    for f in uploaded_files:
        fname = getattr(f, "name", None) or os.path.basename(f)
        club_name = normalize_name(os.path.splitext(fname)[0])
        try:
            if hasattr(f, "read"):
                content = f.read()
                if isinstance(content, str):
                    content_bytes = content.encode('utf-8')
                else:
                    content_bytes = content
            else:
                with open(f, 'rb') as fh:
                    content_bytes = fh.read()
            res = parse_whatsapp_text_file_bytes(content_bytes)
        except Exception as e:
            res = {'total_msgs':0,'unique_senders':0,'event_mentions':0}
        rows.append({'club_name': club_name,
                     'whatsapp_msgs': res['total_msgs'],
                     'whatsapp_unique_senders': res['unique_senders'],
                     'whatsapp_event_mentions': res['event_mentions']})
    return pd.DataFrame(rows)




# Event log parsing


def parse_event_log(file_like):
    if isinstance(file_like, str) or hasattr(file_like, "read"):
        try:
            df = pd.read_csv(file_like)
        except Exception:
            df = pd.read_excel(file_like)
    else:
        df = pd.read_csv(file_like)
    df.columns = [c.strip() for c in df.columns]


    df['club_name'] = df['club_name'].apply(normalize_name)
    if 'club_name' not in df.columns:
        for alt in ['Club Name','club','clubname']:
            if alt in df.columns:
                df = df.rename(columns={alt:'club_name'})
                break
    if 'event_title' not in df.columns:
        for alt in ['Event Title','title','event']:
            if alt in df.columns:
                df = df.rename(columns={alt:'event_title'})
                break
    if 'event_description' not in df.columns:
        for alt in ['Event Description','description','details']:
            if alt in df.columns:
                df = df.rename(columns={alt:'event_description'})
                break


    df['event_title'] = df.get('event_title','').fillna('')
    df['event_description'] = df.get('event_description','').fillna('')


    agg = df.groupby('club_name').agg(
        event_count = ('event_title','count'),
        events_text = ('event_title', lambda s: " ".join(s.tolist())),
        descriptions_text = ('event_description', lambda s: " ".join(s.tolist()))
    ).reset_index()


    agg['text_for_grouping'] = (
        agg['events_text'].fillna('') + ' ' + agg['descriptions_text'].fillna('')
    ).str.lower()


    return df, agg[['club_name','event_count','text_for_grouping']]



# Sentiment per club


def compute_sentiment_per_club(agg_survey_df):
    sia = SentimentIntensityAnalyzer()
    rows = []
    for _, r in agg_survey_df.iterrows():
        txt = r.get('feedback_concat', "") or ""
        if not txt or str(txt).strip() == "":
            score01 = 0.5
        else:
            comp = sia.polarity_scores(str(txt))['compound']
            score01 = (comp + 1.0)/2.0
        rows.append({'club_name': r['club_name'], 'sentiment_score': score01})
    return pd.DataFrame(rows)




# Auto-assign group using keywords


def assign_club_to_category(club_name, text):
    text = f"{club_name} {text}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    # fallback
    return 'others'



# Combine and scoring - IMPROVED VERSION


def compute_group_scores(survey_agg_df, whatsapp_df, sentiment_df, event_agg_df, default_weights=None):
    df = pd.merge(survey_agg_df, sentiment_df, on='club_name', how='outer')
    df = pd.merge(df, whatsapp_df, on='club_name', how='outer')
    df = pd.merge(df, event_agg_df, on='club_name', how='outer')


    df['heard_often_mean'] = df['heard_often_mean'].fillna(0)
    df['participation_mean'] = df['participation_mean'].fillna(0)
    df['feedback_concat'] = df['feedback_concat'].fillna('')
    df['sentiment_score'] = df['sentiment_score'].fillna(0.5)
    df['whatsapp_msgs'] = df['whatsapp_msgs'].fillna(0)
    df['whatsapp_event_mentions'] = df.get('whatsapp_event_mentions', 0).fillna(0)
    df['whatsapp_unique_senders'] = df.get('whatsapp_unique_senders', 0).fillna(0)
    df['event_count'] = df['event_count'].fillna(0)
    df['text_for_grouping'] = df['text_for_grouping'].fillna('')


    # Assign named groups
    df['group'] = df.apply(lambda r: assign_club_to_category(r['club_name'], r['text_for_grouping']), axis=1)


    if default_weights is None:
        default_weights = {'heard':0.30,'participation':0.30,'sentiment':0.20,'whatsapp_msgs':0.10,'event_count':0.10}


    # STEP 1: Calculate group scores (within-group comparison) - UNCHANGED
    final_rows = []
    for group, gdf in df.groupby('group'):
        g = gdf.copy()
        g['heard_norm'] = g['heard_often_mean'] / 5.0
        if g['participation_mean'].max() - g['participation_mean'].min() > 0:
            g['participation_norm'] = (g['participation_mean'] - g['participation_mean'].min()) / (g['participation_mean'].max() - g['participation_mean'].min())
        else:
            g['participation_norm'] = 0.0
        g['sentiment_norm'] = g['sentiment_score']
        if g['whatsapp_msgs'].max() - g['whatsapp_msgs'].min() > 0:
            g['whatsapp_msgs_norm'] = (g['whatsapp_msgs'] - g['whatsapp_msgs'].min()) / (g['whatsapp_msgs'].max() - g['whatsapp_msgs'].min())
        else:
            g['whatsapp_msgs_norm'] = 0.0
        if g['event_count'].max() - g['event_count'].min() > 0:
            g['event_count_n'] = (g['event_count'] - g['event_count'].min()) / (g['event_count'].max() - g['event_count'].min())
        else:
            g['event_count_n'] = 0.0


        w = default_weights
        g['group_score'] = (
            w['heard'] * g['heard_norm'] +
            w['participation'] * g['participation_norm'] +
            w['sentiment'] * g['sentiment_norm'] +
            w['whatsapp_msgs'] * g['whatsapp_msgs_norm'] +
            w['event_count'] * g['event_count_n']
        )
        final_rows.append(g)


    final = pd.concat(final_rows, ignore_index=True)
    
    # STEP 2: Calculate overall scores (global comparison) - IMPROVED
    # Global normalization across ALL clubs for overall score
    final['heard_norm_global'] = final['heard_often_mean'] / 5.0
    
    if final['participation_mean'].max() > 0:
        final['participation_norm_global'] = final['participation_mean'] / final['participation_mean'].max()
    else:
        final['participation_norm_global'] = 0.0
        
    final['sentiment_norm_global'] = final['sentiment_score']
    
    if final['whatsapp_msgs'].max() > 0:
        final['whatsapp_msgs_norm_global'] = final['whatsapp_msgs'] / final['whatsapp_msgs'].max()
    else:
        final['whatsapp_msgs_norm_global'] = 0.0
        
    if final['event_count'].max() > 0:
        final['event_count_norm_global'] = final['event_count'] / final['event_count'].max()
    else:
        final['event_count_norm_global'] = 0.0

    # Calculate overall score using global normalization
    w = default_weights
    final['overall_score'] = (
        w['heard'] * final['heard_norm_global'] +
        w['participation'] * final['participation_norm_global'] +
        w['sentiment'] * final['sentiment_norm_global'] +
        w['whatsapp_msgs'] * final['whatsapp_msgs_norm_global'] +
        w['event_count'] * final['event_count_norm_global']
    )

    final['overall_rank'] = final['overall_score'].rank(method='min', ascending=False).astype(int)

    winners = final.loc[final.groupby('group')['group_score'].idxmax()][['group','club_name','group_score','overall_score']].reset_index(drop=True)
    final = final.sort_values(['group','group_score'], ascending=[True, False])

    # RENAME COLUMNS TO MORE USER-FRIENDLY NAMES
    final = final.rename(columns={
        'heard_often_mean': 'popularity_score',
        'participation_mean': 'participation_score', 
        'sentiment_score': 'sentiment_score',  # already good
        'whatsapp_msgs': 'engagement_score',
        'event_count': 'activity_score',
        'group_score': 'category_score',
        'overall_score': 'overall_score',  # already good
        'overall_rank': 'overall_rank'  # already good
    })

    winners = winners.rename(columns={
        'group_score': 'category_score'
    })

    return final, winners
