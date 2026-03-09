# Ganitha Saviya National Program - AI-Driven Data Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

An intelligent data processing and analytics pipeline for the **Ganitha Saviya National Program 2024-25**, transforming raw program data into actionable insights through AI models, resource forecasting, and network analysis.

**Serving 107,894 Students | Processing 2,096 Seminars | 26 Districts**

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Data Scope](#-data-scope)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Outputs](#-outputs)
- [Technologies](#-technologies)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

The **Ganitha Saviya National Program** is a comprehensive educational initiative designed to deliver mathematics seminars across Sri Lanka. This repository contains an intelligent data pipeline that:

- **Consolidates** real-time data from Google Forms and historical CSV records
- **Cleans and normalizes** 2,096+ seminar records with advanced data validation
- **Predicts** resource requirements using machine learning (XGBoost)
- **Identifies** at-risk volunteers using risk analytics
- **Analyzes** geographic demand and network patterns
- **Generates** interactive dashboards with real-time analytics

### Program Goals

✅ Democratize mathematics education across all 26 districts of Sri Lanka  
✅ Engage qualified volunteers and educators in seminar delivery  
✅ Provide data-driven insights for program optimization  
✅ Forecast resource requirements for scalability  
✅ Monitor volunteer engagement and identify retention risks  

---

## ⭐ Key Features

| Feature | Description |
|---------|-------------|
| 🔗 **Live Data Integration** | Real-time sync with Google Sheets API + historical CSV consolidation |
| 🧹 **Advanced Data Cleaning** | Mojibake detection, date standardization, text normalization, missing value handling |
| 🤖 **Machine Learning Models** | XGBoost for resource forecasting, volunteer risk assessment, network demand analysis |
| 🧠 **AI-Powered Analysis** | Google Gemini API integration for NLP-based insights and text classification |
| 📊 **Interactive Dashboards** | HTML-based visualization with real-time analytics and predictive insights |
| 📈 **Predictive Analytics** | Forecast volunteer requirements, resource allocation, and seminar demand |
| 🌐 **Network Analysis** | Geographic and organizational network mapping using NetworkX |
| 🔐 **Secure Integration** | Service account authentication with environment variable support |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│            Ganitha Saviya Data Pipeline Flow                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │ Google Forms │         │  Historical  │                 │
│  │  (Live API)  │         │   CSV Data   │                 │
│  └──────┬───────┘         └──────┬───────┘                 │
│         │                        │                          │
│         └────────────┬───────────┘                          │
│                      ▼                                      │
│      ┌───────────────────────────┐                         │
│      │ Data Consolidation &      │                         │
│      │ Quality Validation        │                         │
│      └────────────┬──────────────┘                         │
│                   ▼                                         │
│      ┌───────────────────────────┐                         │
│      │ Clean Dataset (2,096)     │                         │
│      │ Ready for Analysis        │                         │
│      └────────┬──────────────────┘                         │
│               │                                            │
│   ┌───────────┼─────────────┬──────────────────┐          │
│   ▼           ▼             ▼                  ▼           │
│ ┌─────────┐ ┌─────────┐ ┌──────────┐  ┌──────────────┐    │
│ │Resource │ │Volunteer│ │ Network  │  │  Gemini NLP  │    │
│ │Forecast │ │  Risk   │ │ Demand   │  │  Analysis    │    │
│ │(XGBoost)│ │Analysis │ │Analysis  │  │ (API)        │    │
│ └────┬────┘ └────┬────┘ └────┬─────┘  └──────┬────────┘    │
│      │           │           │              │              │
│      └───────────┴───────────┴──────────────┘              │
│                      ▼                                      │
│       ┌──────────────────────────┐                         │
│       │  Insights & Predictions  │                         │
│       │  (JSON Output)           │                         │
│       └──────────────┬───────────┘                         │
│                      ▼                                      │
│    ┌─────────────────────────────────┐                    │
│    │ Interactive HTML Dashboard      │                    │
│    │ & Analytics Reports            │                    │
│    └─────────────────────────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Scope

| Metric | Value |
|--------|-------|
| **Total Seminars** | 2,096 |
| **Student Reach** | 107,894 |
| **Geographic Coverage** | 26 Districts |
| **Data Period** | 2024-25 Academic Year |
| **Data Sources** | Google Forms + Historical CSV |
| **Processing Models** | 3 ML Models + 1 NLP API |

### Data Quality Measures

- ✅ Automatic mojibake (corrupted encoding) detection and removal
- ✅ Date standardization with dayfirst format support
- ✅ Text normalization (Title Case, whitespace cleanup)
- ✅ Missing value handling with strategic imputation
- ✅ Duplicate record identification
- ✅ Multi-encoding fallback support (UTF-8, Latin1)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Sheets API credentials
- Gemini API key (for AI-powered analysis)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gs_pipeline.git
   cd gs_pipeline
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up credentials** (see [Configuration](#-configuration) section)

---

## ⚙️ Configuration

### Google Sheets Setup

1. Create a service account in [Google Cloud Console](https://console.cloud.google.com/)
2. Download the service account JSON file
3. Share your Google Sheet with the service account email

**Two configuration methods:**

#### Method 1: Environment Variables (Recommended for CI/CD)
```bash
# Set these environment variables:
export GEMINI_API_KEY="your-gemini-api-key"
export GOOGLE_SHEETS_CREDENTIALS_JSON='{"type":"service_account",...}'
```

#### Method 2: Local Files (Development)
```bash
# Place your service account JSON in the project root
service_account.json
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for NLP analysis | ✅ Yes |
| `GOOGLE_SHEETS_CREDENTIALS_JSON` | Service account credentials (JSON string) | ⚠️ Optional* |

*If not set, the script will look for `service_account.json` file

---

## 📖 Usage

### Run the Pipeline

```bash
python process_data.py
```

### What the Script Does

1. **Fetches** live data from Google Sheets
2. **Loads** historical CSV data
3. **Consolidates** and cleans all data
4. **Applies** ML models for predictions
5. **Performs** network analysis
6. **Generates** dashboard and report JSON files
7. **Outputs** insights and visualizations

### Output Files Generated

- `dashboard_data.json` - Real-time dashboard metrics
- `reports_data.json` - Detailed analysis reports
- `preview/AI.html` - Interactive HTML dashboard

---

## 📁 Project Structure

```
gs_pipeline/
├── README.md                                    # This file
├── about.md                                     # Detailed project documentation
├── process_data.py                              # Main data pipeline script
├── requirements.txt                             # Python dependencies
├── service_account.json                         # Google credentials (not in repo)
├── Ganitha Saviya...Form responses.csv          # Historical data
├── dashboard_data.json                          # Output: Dashboard data
├── reports_data.json                            # Output: Analysis reports
├── preview/
│   └── AI.html                                  # Output: Interactive dashboard
└── .github/                                     # GitHub workflows and configs
```

---

## 📤 Outputs

### Dashboard Data (`dashboard_data.json`)
Real-time metrics for interactive visualization:
- District-wise seminar counts
- Student enrollment statistics
- Volunteer allocation data
- Resource availability
- Attendance trends

### Reports Data (`reports_data.json`)
Comprehensive analysis outputs:
- Resource forecasting predictions
- Volunteer risk assessments
- Network demand analysis
- Geographic distribution insights
- AI-powered recommendations

### Interactive Dashboard (`preview/AI.html`)
Visual analytics interface with:
- Real-time seminar tracking
- District-wise performance metrics
- Volunteer engagement dashboards
- Predictive analytics charts
- Network visualization

---

## 🛠️ Technologies

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **xgboost** - Machine learning (resource forecasting)
- **scikit-learn** - ML utilities and preprocessing
- **networkx** - Network analysis and graphs
- **gspread** - Google Sheets API client
- **google-auth** - Google API authentication
- **google-generativeai** - Gemini API for NLP

### Data & APIs
- Google Sheets API (live data)
- Google Generative AI API (Gemini)
- CSV file processing

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- 🐛 Bug fixes and data validation improvements
- ✨ New ML models for better predictions
- 📊 Enhanced dashboard visualizations
- 📚 Documentation and tutorials
- 🤖 Additional AI-powered analysis features

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Support & Contact

For questions, issues, or suggestions:
- 📧 Open an issue on GitHub
- 💬 Check the [about.md](about.md) for detailed project documentation
- 🔍 Review [process_data.py](process_data.py) for implementation details

---

## 🙏 Acknowledgments

- **Ganitha Saviya National Program** team for the initiative
- All volunteers and educators contributing to mathematics education
- Google for the APIs and infrastructure support

---

<div align="center">

**Made with ❤️ for mathematics education in Sri Lanka**

[⬆ Back to top](#ganitha-saviya-national-program---ai-driven-data-pipeline)

</div>
