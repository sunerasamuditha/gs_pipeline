import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import json
import os
import google.generativeai as genai
from datetime import datetime, timedelta
import networkx as nx
from xgboost import XGBRegressor
import re

# --- CONFIGURATION ---
SHEET_NAME = "Ganitha Saviya National Program 2025/26 (Responses)"
CSV_PATH = "Ganitha Saviya National Program 2024_25 (Responses) - Form responses.csv"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def fetch_data():
    print("Connecting to Google Sheets...")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    if os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON"):
        creds_json = json.loads(os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON"))
        if 'private_key' in creds_json:
             creds_json['private_key'] = creds_json['private_key'].replace('\\n', '\n')
        creds = Credentials.from_service_account_info(creds_json, scopes=scope)
    elif os.path.exists("service_account.json"):
        creds = Credentials.from_service_account_file("service_account.json", scopes=scope)
    else:
        raise Exception("Google Sheets credentials not found.")

    client = gspread.authorize(creds)
    spreadsheet = client.open(SHEET_NAME)
    
    try:
        sheet = spreadsheet.worksheet("Form responses 1")
    except gspread.exceptions.WorksheetNotFound:
        print("Warning: 'Form responses 1' not found, using first sheet.")
        sheet = spreadsheet.sheet1
    
    all_values = sheet.get_all_values()
    
    if all_values:
        headers = all_values[0]
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

    # Try UTF-8 first, fallback to Latin-1 only if necessary
    # Note: The fallback is often what creates the "Mojibake" (À¶½...) errors
    try:
        historical_data = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    except Exception as e:
        print(f"Warning: CSV read failed with utf-8-sig, trying latin1 fallback. Error: {e}")
        historical_data = pd.read_csv(CSV_PATH, encoding='latin1')

    combined_df = pd.concat([historical_data, live_data], ignore_index=True)
    return combined_df

def remove_corrupted_rows(df):
    """
    Detects and removes rows containing encoding errors (Mojibake).
    Pattern: Characters in Latin-1 Supplement range (0x80-0xFF) 
    which appear when UTF-8 (Sinhala/Tamil) is misread as Latin-1.
    """
    initial_count = len(df)
    
    # Columns to check for corruption
    check_cols = ['Name of the School ', 'District', 'Type of Seminar', 'Medium']
    # Only check columns that actually exist
    check_cols = [c for c in check_cols if c in df.columns]

    def is_corrupted(text):
        if not isinstance(text, str): return False
        
        # 1. Normalize: Replace Non-Breaking Space (\u00A0) with normal space 
        # (NBSP is valid, we don't want to delete rows because of it)
        text = text.replace('\u00A0', ' ')
        
        # 2. Check for Replacement Character ()
        if '\ufffd' in text: return True
        
        # 3. Check for Mojibake Pattern (High-ASCII junk like À, ¶, ½)
        # Sinhala/Tamil are > 0x0800, so they won't trigger this.
        # English is < 0x0080. 
        # This range (0x80 - 0xFF) captures the specific corruption you saw.
        if re.search(r'[\u0080-\u00FF]', text):
            return True
        return False

    # Apply filter
    mask = df[check_cols].applymap(is_corrupted).any(axis=1)
    df_clean = df[~mask].copy()
    
    dropped_count = initial_count - len(df_clean)
    if dropped_count > 0:
        print(f"⚠️ Removed {dropped_count} rows containing encoding errors (Mojibake).")
        
    return df_clean

def clean_data(df):
    print(f"Cleaning data... Initial count: {len(df)}")
    
    # 0. PRE-CLEAN: Remove corrupted encoding rows
    df = remove_corrupted_rows(df)
    
    # 1. Standardize Dates
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # 2. Text Normalization
    text_columns = ['District', 'Name of the School ', 'Medium', 'Type of Seminar']
    for col in text_columns:
        if col in df.columns:
            # Strip, Title Case, and remove accidental non-breaking spaces
            df[col] = df[col].astype(str).str.strip().str.replace('\u00A0', ' ').str.title()

    # 3. Volunteer Parsing
    volunteer_cols = [c for c in df.columns if 'Sasnaka Sansada member' in c]
    volunteer_col = volunteer_cols[0] if volunteer_cols else None

    def extract_names(entry):
        if pd.isna(entry): return []
        names = []
        for line in str(entry).split('\n'):
            clean = line.split('.', 1)[-1].strip()
            if clean: names.append(clean)
        return names

    if volunteer_col:
        df['clean_volunteers'] = df[volunteer_col].apply(extract_names)
    else:
        df['clean_volunteers'] = [[] for _ in range(len(df))]

    student_col = 'Number of Students participated'
    df[student_col] = pd.to_numeric(df[student_col], errors='coerce').fillna(0)
    
    print(f"Data cleaned. Rows remaining: {len(df)}")
    return df

# --- REPORT GENERATOR EXPORT ---
def export_report_data(df):
    print("Exporting data for Report Generator...")
    
    # Select safe, public columns
    safe_columns = [
        'Date', 'District', 'Name of the School ', 'Type of Seminar', 
        'Medium', 'Number of Students participated'
    ]
    
    existing_cols = [c for c in safe_columns if c in df.columns]
    report_df = df[existing_cols].copy()
    
    report_df['Date'] = report_df['Date'].dt.strftime('%Y-%m-%d')
    
    if 'clean_volunteers' in df.columns:
        report_df['Volunteers'] = df['clean_volunteers'].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    
    # Force UTF-8 encoding to prevent new errors
    with open('reports_data.json', 'w', encoding='utf-8') as f:
        json.dump(report_df.to_dict(orient='records'), f, indent=4, ensure_ascii=False)
        
    print(f"Success! reports_data.json created with {len(report_df)} records.")

# --- AI MODEL 1: RESOURCE FORECASTER ---
def run_resource_forecaster(df):
    print("Running AI Model 1: Resource Forecaster...")
    try:
        model_df = df.copy()
        model_df = model_df.dropna(subset=['Date', 'District'])
        
        if len(model_df) < 10: return {}

        model_df['Month'] = model_df['Date'].dt.month
        model_df['Is_Exam_Season'] = model_df['Month'].isin([5, 8, 12]).astype(int)
        
        X = model_df[['Month', 'Is_Exam_Season', 'District']]
        X = pd.get_dummies(X, columns=['District'], drop_first=False)
        y = model_df['Number of Students participated']

        model = XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
        model.fit(X, y)

        next_month = (datetime.now() + timedelta(days=30)).month
        is_exam = 1 if next_month in [5, 8, 12] else 0
        
        forecasts = {}
        top_districts = df['District'].value_counts().head(5).index.tolist()

        for district in top_districts:
            input_data = {'Month': [next_month], 'Is_Exam_Season': [is_exam]}
            for col in X.columns:
                if col.startswith('District_'):
                    dist_name = col.replace('District_', '')
                    input_data[col] = [1 if dist_name == district else 0]
            
            input_df = pd.DataFrame(input_data).reindex(columns=X.columns, fill_value=0)
            pred_students = max(0, int(model.predict(input_df)[0]))
            
            forecasts[district] = {
                "predicted_students": pred_students,
                "paper_sheets_needed": int(pred_students * 5 * 1.15)
            }
        return forecasts
    except Exception as e:
        print(f"Resource Forecaster Error: {e}")
        return {}

# --- AI MODEL 2: VOLUNTEER RISK ---
def run_volunteer_risk_model(df):
    print("Running AI Model 2: Volunteer Risk...")
    try:
        vol_data = []
        for _, row in df.iterrows():
            if pd.isna(row['Date']): continue
            for vol in row['clean_volunteers']:
                vol_data.append({'Name': vol, 'Date': row['Date']})
        
        v_df = pd.DataFrame(vol_data)
        if v_df.empty: return []

        current_date = datetime.now()
        risky_volunteers = []

        for name, group in v_df.groupby('Name'):
            group = group.sort_values('Date')
            last_active = group['Date'].max()
            days_inactive = (current_date - last_active).days
            recent_events = group[group['Date'] > (current_date - timedelta(days=30))]
            risk_level = "Low"
            reason = ""

            if days_inactive > 90:
                risk_level = "High (Churn)"
                reason = "Inactive for >90 days"
            elif len(recent_events) >= 4:
                risk_level = "Medium (Burnout)"
                reason = f"{len(recent_events)} seminars in last 30 days"

            if risk_level != "Low":
                risky_volunteers.append({
                    "name": name,
                    "risk_level": risk_level,
                    "reason": reason,
                    "last_active": last_active.strftime('%Y-%m-%d')
                })

        return sorted(risky_volunteers, key=lambda x: x['last_active'])[:10]
    except Exception as e:
        print(f"Volunteer Risk Error: {e}")
        return []

# --- AI MODEL 3: DEMAND PROPAGATION ---
def run_demand_model(df):
    print("Running AI Model 3: Network Demand...")
    try:
        G = nx.Graph()
        for _, row in df.iterrows():
            school_raw = row.get('Name of the School ', '')
            if pd.isna(school_raw) or str(school_raw).strip() == '': continue
            
            school = str(school_raw).strip()
            G.add_node(school, type='school')
            
            for vol in row['clean_volunteers']:
                G.add_node(vol, type='volunteer')
                G.add_edge(school, vol)

        centrality = nx.degree_centrality(G)
        predictions = []
        schools = [n for n, d in G.nodes(data=True) if d.get('type') == 'school']
        
        for school in schools:
            neighbors = [n for n in G.neighbors(school) if G.nodes[n].get('type') == 'volunteer']
            if not neighbors: continue
            
            score = np.mean([centrality[n] for n in neighbors]) * 1000 
            predictions.append({"school": school, "demand_score": round(score, 2)})
            
        return sorted(predictions, key=lambda x: x['demand_score'], reverse=True)[:5]
    except Exception as e:
        print(f"Demand Model Error: {e}")
        return []

# --- GEMINI API ---
def run_remarks_analysis(df):
    print("Running Gemini AI for Remarks...")
    remarks_col = "Any remarks on the school or seminar."
    if remarks_col not in df.columns: return "No remarks column found."

    remarks_series = df[remarks_col].dropna().astype(str)
    remarks_series = remarks_series[remarks_series.str.len() > 5]
    all_remarks = remarks_series.tolist()
    
    if len(all_remarks) > 50: all_remarks = all_remarks[-50:]

    insights_text = "No insights available."
    if GEMINI_API_KEY and all_remarks:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"Analyze these seminar remarks and give 10 bullet points on key operational issues or praise: {all_remarks}"
            response = model.generate_content(prompt)
            insights_text = response.text
        except Exception as e:
            print(f"NLP Failed: {e}")
            
    return insights_text

def main():
    try:
        df = fetch_data()
        df = clean_data(df)

        export_report_data(df)

        resource_forecast = run_resource_forecaster(df)
        volunteer_risks = run_volunteer_risk_model(df)
        demand_predictions = run_demand_model(df)
        nlp_insights = run_remarks_analysis(df)

        output = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "total_seminars": len(df),
            "total_students": int(df['Number of Students participated'].sum()),
            "district_breakdown": df['District'].value_counts().to_dict(),
            "ai_resource_forecast": resource_forecast,
            "ai_volunteer_risks": volunteer_risks,
            "ai_demand_predictions": demand_predictions,
            "ai_remarks_insights": nlp_insights
        }

        with open("dashboard_data.json", "w", encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print("Success! dashboard_data.json created with 4 AI models.")
        
    except Exception as e:
        print(f"CRITICAL ERROR in Main: {e}")

if __name__ == "__main__":
    main()