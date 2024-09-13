# MLOps Rakuten2

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Description
MLOps Rakuten2 is a machine learning operations project designed to streamline the deployment and management of machine learning models. This project includes data preprocessing, model training, and API deployment using Docker and Airflow.

## Installation

### Prerequisites
- Docker
- Docker Compose
- Python 3.9
- Git

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/mlops_rakuten2.git
   cd mlops_rakuten2
2. Install Docker and Kubernetes (Docker desktop for Windows)
3. Install python dependencies:
    pip install -r requirements.txt

## Usage

### Pre-requisite:
#### Create a model
#### Build and Push docker images to DockerHub

### Start application with Kubernetes

### Start API and/or Streamlit localy

## Project Structure
mlops_rakuten2/
├── dags/
│   ├── data_preprocess.py
│   └── models/
│       └── modelsClass.py
├── data/
│   ├── raw/
│   │   ├── X_train_update.csv
│   │   └── Y_train_CVw08PX.csv
│   └── clean/
│       └── silverData_vgg16.csv
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── pretrain_models/
│   ├── gold_vgg16.h5
│   ├── candidate_vgg16.h5
│   └── encoder.joblib
├── streamlit/
│   ├── app.py
│   ├── pages/
│   │   ├── page1.py
│   │   └── page2.py
│   └── requirements.txt
├── .env
├── .gitignore
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
├── README.md
└── LICENSE

## Contributing

## License
NONE