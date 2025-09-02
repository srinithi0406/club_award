# Club Awards Ranking System

## Overview
This is a Student Club Activity Monitoring and Awarding System developed for the World Trade College. It helps management evaluate and rank clubs based on various parameters including:

- Survey data collected from students (perceived popularity and participation)
- WhatsApp group chat activity per club  
- Academic calendar event logs
- Sentiment analysis of student feedback

The system automatically groups clubs with similar activities and spirit, calculates multi-factor scores, and generates both category-wise and overall club rankings. The UI is built using Streamlit for easy accessibility.

## Features
- Import survey CSV, WhatsApp chat exports, and event logs
- Automatic grouping of clubs based on keywords
- Calculation of normalized scores for popularity, participation, sentiment, engagement, and activity
- Combined scoring algorithm using weighted averages
- Ranking and identifying category-wise and overall top clubs
- Dashboard-style visualization with quick insights
- Downloadable CSV reports of rankings and winners

## Installation
1. Clone the repository
2. Setup Python environment (Python 3.7+)
3. Install required packages:

pip install -r requirements.txt

4. Ensure nltk data is downloaded (done automatically in code)

## Running the App
Run the Streamlit app using:


## Using the App
1. Upload your survey CSV file containing club names, ratings, and participation data
2. Upload the event log CSV/XLSX showing club events across the academic year
3. Upload WhatsApp chat exports (.txt format) for each club's group
4. Click "Process and Compute"
5. View category-wise winners and all clubs rankings
6. Download filtered rankings and winners as CSV files

## File Format Requirements

### Survey CSV
Must contain columns:
- `club_name` (required)
- `heard_often` (1-5 scale) or alternatives: `awareness`, `how_often`, `heard`
- `participation_count` (integer) or `participated` (yes/no)
- `feedback_text` (optional) or alternatives: `review_text`, `review`, `feedback`, `comments`

### Event Log CSV/XLSX
Must contain columns:
- `club_name` (required)
- `event_title` (required)
- `event_description` (optional)
- `date` (optional)

### WhatsApp Chat Exports
- One .txt file per club
- Standard WhatsApp export format
- File name should match club name (e.g., "coding club.txt")

## Scoring Methodology
- **Popularity Score**: Based on survey awareness ratings (0-1 scale)
- **Participation Score**: Based on actual participation rates (0-1 scale) 
- **Sentiment Score**: VADER sentiment analysis on feedback text (0-1 scale)
- **Engagement Score**: WhatsApp message activity normalized globally (0-1 scale)
- **Activity Score**: Number of events normalized globally (0-1 scale)

**Overall Score**: Weighted combination using default weights:
- Heard Often: 30%
- Participation: 30% 
- Sentiment: 20%
- WhatsApp Messages: 10%
- Event Count: 10%

## Club Grouping
Clubs are automatically grouped into categories using keyword matching:
- **Tech**: coding, robotics, programming, hackathon, etc.
- **Sports**: football, basketball, cricket, tennis, training, etc.
- **Entertainment**: music, dance, drama, theatre, performance, etc.
- **Literature & Knowledge**: quiz, debate, tamil, lecture, cultural, etc.
- **Others**: Fallback for unmatched clubs

## Output
- **Category Winners**: Best club in each group based on within-group comparison
- **Overall Rankings**: All clubs ranked globally by overall score
- **Quick Insights**: Total clubs, most popular, most active
- **CSV Downloads**: Filtered data matching displayed tables

## Notes
- Club grouping uses keyword-based matching on event descriptions and club names
- Popularity scores and participation ratings depend on survey response quality
- WhatsApp chat analytics include message count and engagement metrics
- Event log parses event frequency and collaboration patterns
- Scores are normalized globally for fair cross-category comparison
- Global normalization prevents small group bias (e.g., robotics club issue)

## Future Enhancements
- Integration with Instagram for social media analytics
- Enhanced NLP for sentiment and topic analysis on chat/feedback
- Voting system integration with bias correction
- Machine learning-based club similarity detection
- Real-time data integration and monitoring

