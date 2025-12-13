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
from sklearn.model_selection import train_test_split

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
    
    # Try to open 'Form responses 1', fallback to first sheet if not found
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

    historical_data = pd.read_csv(CSV_PATH)
    combined_df = pd.concat([historical_data, live_data], ignore_index=True)
    return combined_df

def clean_data(df):
    print(f"Cleaning data... Initial count: {len(df)}")
    # Assuming DD/MM/YYYY format based on typical regional usage
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Identify Volunteer Column safely
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

# --- AI MODEL 1: RESOURCE FORECASTER (XGBoost) ---
def run_resource_forecaster(df):
    print("Running AI Model 1: Resource Forecaster...")
    try:
        # 1. Feature Engineering
        model_df = df.copy()
        initial_count = len(model_df)
        model_df = model_df.dropna(subset=['Date', 'District'])
        dropped_count = initial_count - len(model_df)
        if dropped_count > 0:
            print(f"  - Note: {dropped_count} rows dropped due to missing Date/District for forecasting.")
        
        model_df['Month'] = model_df['Date'].dt.month
        # Exam Season in SL: May, August, December usually
        model_df['Is_Exam_Season'] = model_df['Month'].isin([5, 8, 12]).astype(int)
        
        # Prepare Data for Training
        X = model_df[['Month', 'Is_Exam_Season', 'District']]
        # One-Hot Encode District
        X = pd.get_dummies(X, columns=['District'], drop_first=False)
        y = model_df['Number of Students participated']

        if len(X) < 10: return {} # Not enough data

        # Train XGBoost
        model = XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
        model.fit(X, y)

        # 2. Predict for Next Month
        next_month_date = datetime.now() + timedelta(days=30)
        next_month = next_month_date.month
        is_exam = 1 if next_month in [5, 8, 12] else 0
        
        forecasts = {}
        # Predict for top 5 active districts
        top_districts = df['District'].value_counts().head(5).index.tolist()

        for district in top_districts:
            # Create input row
            input_data = {
                'Month': [next_month],
                'Is_Exam_Season': [is_exam]
            }
            # Add district dummy columns (set 1 for current district, 0 for others)
            for col in X.columns:
                if col.startswith('District_'):
                    dist_name = col.replace('District_', '')
                    input_data[col] = [1 if dist_name == district else 0]
            
            # Align columns with training data
            input_df = pd.DataFrame(input_data)
            # Ensure order matches training
            input_df = input_df.reindex(columns=X.columns, fill_value=0)
            
            pred_students = max(0, int(model.predict(input_df)[0]))
            
            forecasts[district] = {
                "predicted_students": pred_students,
                "paper_sheets_needed": int(pred_students * 5 * 1.15) # 5 sheets + 15% buffer
            }
            
        return forecasts
    except Exception as e:
        print(f"Resource Forecaster Error: {e}")
        return {}

# --- AI MODEL 2: VOLUNTEER RISK (Heuristic/Behavioral) ---
def run_volunteer_risk_model(df):
    print("Running AI Model 2: Volunteer Risk...")
    try:
        # Explode volunteers to rows
        vol_data = []
        for _, row in df.iterrows():
            if pd.isna(row['Date']): continue
            for vol in row['clean_volunteers']:
                vol_data.append({'Name': vol, 'Date': row['Date']})
        
        v_df = pd.DataFrame(vol_data)
        if v_df.empty: return []

        current_date = datetime.now()
        risky_volunteers = []

        # Analyze each volunteer
        for name, group in v_df.groupby('Name'):
            group = group.sort_values('Date')
            last_active = group['Date'].max()
            days_inactive = (current_date - last_active).days
            total_events = len(group)
            
            # Check for Burnout (Too many events recently)
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

        # Return top 10 risks
        return sorted(risky_volunteers, key=lambda x: x['last_active'])[:10]
    except Exception as e:
        print(f"Volunteer Risk Error: {e}")
        return []

# --- AI MODEL 3: DEMAND PROPAGATION (NetworkX) ---
def run_demand_model(df):
    print("Running AI Model 3: Network Demand...")
    try:
        G = nx.Graph()
        
        # Build Graph: School <-> Volunteer
        for _, row in df.iterrows():
            school_raw = row.get('Name of the School ', '')
            if pd.isna(school_raw) or str(school_raw).strip().lower() == 'nan' or str(school_raw).strip() == '':
                continue
                
            school = str(school_raw).strip()
            G.add_node(school, type='school')
            
            for vol in row['clean_volunteers']:
                G.add_node(vol, type='volunteer')
                G.add_edge(school, vol)

        # Calculate Centrality (Influence)
        centrality = nx.degree_centrality(G)
        
        predictions = []
        schools = [n for n, d in G.nodes(data=True) if d.get('type') == 'school']
        
        for school in schools:
            # Score = Avg centrality of volunteers who visited
            neighbors = [n for n in G.neighbors(school) if G.nodes[n].get('type') == 'volunteer']
            if not neighbors: continue
            
            # Network Score calculation
            score = np.mean([centrality[n] for n in neighbors]) * 1000 
            
            predictions.append({
                "school": school,
                "demand_score": round(score, 2)
            })
            
        # Top 5 schools with strongest network effects
        return sorted(predictions, key=lambda x: x['demand_score'], reverse=True)[:5]
    except Exception as e:
        print(f"Demand Model Error: {e}")
        return []

# --- GEMINI API (Remarks Summary) ---
def run_remarks_analysis(df):
    print("Running Gemini AI for Remarks...")
    remarks_col = "Any remarks on the school or seminar."
    # Filter for non-empty string remarks
    remarks_series = df[remarks_col].dropna().astype(str)
    remarks_series = remarks_series[remarks_series.str.len() > 5] # Filter short junk
    recent_remarks = remarks_series.tail(100).tolist()

    insights_text = "No insights available."
    
    if GEMINI_API_KEY and recent_remarks:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"Analyze these seminar remarks and give 3 bullet points on key operational issues or praise. Be specific: {recent_remarks}"
            response = model.generate_content(prompt)
            insights_text = response.text
        except Exception as e:
            print(f"NLP Failed: {e}")
            
    return insights_text

def main():
    try:
        df = fetch_data()
        df = clean_data(df)

        # Run The 4 Engines
        resource_forecast = run_resource_forecaster(df)
        volunteer_risks = run_volunteer_risk_model(df)
        demand_predictions = run_demand_model(df)
        nlp_insights = run_remarks_analysis(df)

        # Prepare Final Output
        output = {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "total_seminars": len(df),
            "total_students": int(df['Number of Students participated'].sum()),
            "district_breakdown": df['District'].value_counts().to_dict(),
            
            # AI Outputs
            "ai_resource_forecast": resource_forecast,
            "ai_volunteer_risks": volunteer_risks,
            "ai_demand_predictions": demand_predictions,
            "ai_remarks_insights": nlp_insights
        }

        with open("dashboard_data.json", "w") as f:
            json.dump(output, f, indent=4)
        print("Success! dashboard_data.json created with 4 AI models.")
        
    except Exception as e:
        print(f"CRITICAL ERROR in Main: {e}")

if __name__ == "__main__":
    main()