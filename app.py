# app.py
import streamlit as st
import pandas as pd
from src.processors import parse_survey, parse_whatsapp_folder, parse_event_log, compute_sentiment_per_club, compute_group_scores

st.set_page_config(page_title="Club Awards — Auto Grouping Demo", layout="wide")
st.title("Club Awards — Auto Grouping (Survey + WhatsApp + Event Log)")

st.markdown("""
**Upload files to demo:**  
1. Survey CSV (Google Forms export) — must have `club_name` column.  
2. Event log CSV (club event calendar) — columns: `club_name`, `event_title`, `event_description`, `date` (date optional).  
3. WhatsApp .txt exports (one file per club) — multiple files allowed.
""")

survey_file = st.file_uploader("Upload Survey CSV", type=["csv"])
event_log_file = st.file_uploader("Upload Event Log CSV (or XLSX)", type=["csv","xlsx"])
wa_files = st.file_uploader("Upload WhatsApp .txt files (one per club)", type=["txt"], accept_multiple_files=True)

if st.button("Process and Compute"):
    if survey_file is None or event_log_file is None or not wa_files:
        st.error("Please upload survey CSV, event log CSV, and at least one WhatsApp .txt file.")
    else:
        try:
            with st.spinner("Parsing survey..."):
                survey_agg = parse_survey(survey_file)
            st.success(f"Survey parsed: {len(survey_agg)} clubs aggregated.")

            with st.spinner("Parsing event log..."):
                event_df, event_agg = parse_event_log(event_log_file)
            st.success(f"Event log parsed: {len(event_df)} events, {len(event_agg)} clubs in log.")

            with st.spinner("Parsing WhatsApp files..."):
                wa_df = parse_whatsapp_folder(wa_files)
            st.success(f"WhatsApp parsed: {len(wa_df)} club files.")

            with st.spinner("Computing sentiment..."):
                sent_df = compute_sentiment_per_club(survey_agg)
            st.success("Sentiment computed.")

            with st.spinner("Computing group scores and auto-grouping..."):
                final_df, winners_df = compute_group_scores(survey_agg, wa_df, sent_df, event_agg)
                final_df.to_csv("outputs/combined_scores.csv", index=False)
                winners_df.to_csv("outputs/group_winners.csv", index=False)

            st.success("Computed final scores.")

            st.subheader("Auto-grouped winners (one per group)")
            st.dataframe(winners_df)

            st.subheader("All clubs (grouped and scored)")
            show_cols = ['club_name','group','heard_often_mean','participation_mean','sentiment_score','whatsapp_msgs','event_count','group_score','overall_score']
            st.dataframe(final_df[show_cols])

            # overall winner
            top = final_df.sort_values('overall_score', ascending=False).iloc[0]
            st.success(f"🏆 Overall best club: **{top['club_name']}** (overall_score: {top['overall_score']:.3f})")

            st.download_button("Download final scores CSV", final_df.to_csv(index=False), "final_scores.csv", "text/csv")
            st.download_button("Download group winners CSV", winners_df.to_csv(index=False), "group_winners.csv", "text/csv")

        except Exception as e:
            st.exception(e)
