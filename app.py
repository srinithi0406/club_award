# app.py
import streamlit as st
import pandas as pd
from src.processors import parse_survey, parse_whatsapp_folder, parse_event_log, compute_sentiment_per_club, compute_group_scores


st.set_page_config(page_title="Club Awards ", layout="wide")
st.title("Club Awards ")


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

            # UPDATED: Display winners with new column names
            st.subheader("Category Winners (Best club in each category)")
            winners_display_cols = ['group', 'club_name', 'category_score', 'overall_score']
            if 'category_score' in winners_df.columns:
                st.dataframe(winners_df[winners_display_cols])
            else:
                # Fallback if old column names still exist
                st.dataframe(winners_df)

            # UPDATED: Display all clubs with new column names and overall rank
            # CHANGE 1: SORT BY OVERALL RANK
            st.subheader("All Clubs Rankings")
            show_cols = [
                'club_name', 'group', 'popularity_score', 'participation_score', 
                'sentiment_score', 'engagement_score', 'activity_score', 
                'category_score', 'overall_score', 'overall_rank'
            ]
            
            # Check if new column names exist, otherwise fall back to old names
            available_cols = [col for col in show_cols if col in final_df.columns]
            if len(available_cols) < len(show_cols):
                # Fallback to old column names if new ones don't exist
                show_cols = ['club_name','group','heard_often_mean','participation_mean','sentiment_score','whatsapp_msgs','event_count','group_score','overall_score','overall_rank']
                available_cols = [col for col in show_cols if col in final_df.columns]
            
            # Sort by overall rank (ascending - rank 1 first)
            final_df_sorted = final_df.sort_values('overall_rank').reset_index(drop=True)
            st.dataframe(final_df_sorted[available_cols])

            # UPDATED: Overall winner display
            top = final_df.sort_values('overall_score', ascending=False).iloc[0]
            rank_text = f" (Rank #{top.get('overall_rank', 'N/A')})" if 'overall_rank' in final_df.columns else ""
            st.success(f"🏆 Overall Best Club: **{top['club_name']}** - Score: {top['overall_score']:.3f}{rank_text}")

            # Additional insights
            st.subheader("📊 Quick Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Clubs", len(final_df))
            
            with col2:
                # CHANGE 2: HANDLE MULTIPLE CLUBS WITH SAME POPULARITY SCORE
                max_popularity = final_df['popularity_score'].max()
                most_popular_clubs = final_df[final_df['popularity_score'] == max_popularity]['club_name'].tolist()
                
                if len(most_popular_clubs) == 1:
                    st.metric("Most Popular", most_popular_clubs[0])
                else:
                    most_popular_text = ", ".join(most_popular_clubs)
                    st.metric("Most Popular", f"{len(most_popular_clubs)} clubs tied")
                    st.caption(most_popular_text)
            

            # Download buttons
            st.download_button(
                "Download All Clubs Rankings CSV", 
                final_df_sorted[available_cols].to_csv(index=False), 
                "all_clubs_rankings.csv", 
                "text/csv"
            )
            st.download_button(
                "Download Category Winners CSV", 
                winners_df[winners_display_cols].to_csv(index=False), 
                "category_winners.csv", 
                "text/csv"
            )

        except Exception as e:
            st.exception(e)
