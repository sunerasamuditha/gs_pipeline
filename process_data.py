import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import os
import google.generativeai as genai
from datetime import datetime

SHEET_NAME = "Ganitha Saviya National Program 2025/26 (Responses)"
CSV_PATH = "Ganitha Saviya National Program 2024_25 (Responses) - Form responses.csv"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def fetch_data():

    print("Connecting to Google Sheets...")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    if os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON"):
        creds_json = json.loads(os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON"))
        # Fix for potential newline escaping issues in GitHub Secrets
        if 'private_key' in creds_json:
             creds_json['private_key'] = creds_json['private_key'].replace('\\n', '\n')
        creds = Credentials.from_service_account_info(creds_json, scopes=scope)
    elif os.path.exists("service_account.json"):
        creds = Credentials.from_service_account_file("service_account.json", scopes=scope)
    else:
        raise Exception("Google Sheets credentials not found. Set GOOGLE_SHEETS_CREDENTIALS_JSON env var or ensure service_account.json exists.")

    client = gspread.authorize(creds)

    sheet = client.open(SHEET_NAME).sheet1
    
    # Use get_all_values to avoid issues with duplicate/empty headers
    all_values = sheet.get_all_values()
    if all_values:
        headers = all_values[0]
        # Deduplicate headers
        seen = {}
        new_headers = []
        for h in headers:
            if h in seen:
                seen[h] += 1
                new_headers.append(f"{h}.{seen[h]}")
            else:
                seen[h] = 0
                new_headers.append(h)
        headers = new_headers

        data = all_values[1:]
        live_data = pd.DataFrame(data, columns=headers)
    else:
        live_data = pd.DataFrame()

    historical_data = pd.read_csv(CSV_PATH)

    combined_df = pd.concat([historical_data, live_data], ignore_index = True)
    return combined_df

def clean_data(df):

    print("Cleaning data...")

    df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce')

    volunteer_col = [c for c in df.columns if 'Sasnaka Sansada member' in c][0]

    def extract_names(entry):

        if pd.isna(entry): return []
        names = []
        for line in str(entry).split('\n'):
            clean = line.split('.', 1)[-1].strip()
            if clean: names.append(clean)
        return names

    df['clean_volunteers'] = df[volunteer_col].apply(extract_names)

    student_col = 'Number of Students participated'
    df[student_col] = pd.to_numeric(df[student_col], errors = 'coerce').fillna(0)

    return df

def run_analysis(df):

    print("Running AI Models...")

    stats = df['Number of Students participated'].describe()
    mean = stats['mean']
    std = stats['std']

    df['is_anomaly'] = abs(df['Number of Students participated'] - mean) > 2 * std
    anomalies = df[df['is_anomaly']].copy()

    remarks_col = "Any remarks on the school or seminar."
    recent_remarks = df[remarks_col].dropna().tail(20).tolist()

    insights_text = "No insights available."
    
    if GEMINI_API_KEY and recent_remarks:
        try:
            genai.configure(api_key = GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"Analyze these seminar remarks and give 3 bullet points on key operational issues or praise: {recent_remarks}"
            response = model.generate_content(prompt)
            insights_text = response.text
        except Exception as e:
            print(f"NLP Failed: {e}")

    return anomalies, insights_text

def main():

    df = fetch_data()
    df = clean_data(df)
    anomalies, nlp_insights = run_analysis(df)

    if not anomalies.empty:
        anomalies['Date'] = anomalies['Date'].dt.strftime('%Y-%m-%d').fillna('')

    output = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "total_seminars": len(df),
        "total_students": int(df['Number of Students participated'].sum()),
        "nlp_insights": nlp_insights,
        "anomalies": anomalies[['Date', 'Name of the School ', 'Number of Students participated']].to_dict(orient='records'),
        "district_breakdown": df['District'].value_counts().to_dict()
    }

    with open("dashboard_data.json", "w") as f:
        json.dump(output, f, indent=4)
    print("Data processing complete. Output saved to dashboard_data.json")

if __name__ == "__main__":
    main()