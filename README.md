# Bank Marketing MLOps Pipeline

End-to-end MLOps pipeline for predicting bank term deposit subscriptions using Apache Airflow, FastAPI, and Streamlit.

## Features

- Automated ML pipeline orchestrated with Apache Airflow
- REST API service with FastAPI
- Interactive Streamlit dashboard
- Exploratory data analysis notebook

## Tech Stack

- **Orchestration**: Apache Airflow
- **API**: FastAPI
- **Frontend**: Streamlit
- **ML**: Scikit-learn, Pandas

## Project Structure

```
├── api/                    # FastAPI prediction service
├── dags/                   # Airflow pipeline DAGs
├── streamlit/              # Streamlit dashboard
├── EDA(Business Analysis).ipynb
├── airflow.cfg
└── webserver_config.py
```

## Installation

```bash
# Clone repository
git clone https://github.com/Bishnu-alt/Bank-Marketing-MLops-full-pipeline.git
cd Bank-Marketing-MLops-full-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize Airflow
airflow db init
```

## Usage

**Start Airflow:**
```bash
airflow webserver --port 8080
airflow scheduler  # In new terminal
```

**Run API:**
```bash
cd api
uvicorn main:app --reload --port 8000
```

**Launch Dashboard:**
```bash
cd streamlit
streamlit run app.py
```

## Pipeline Workflow

1. Data Ingestion
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Deployment

## License

MIT License

## Author

[@Bishnu-alt](https://github.com/Bishnu-alt)
