FROM docker.io/apache/airflow:latest-python3.9
COPY requirementsDockerImage.txt /
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirementsDockerImage.txt

# Install Docker CLI
USER root
RUN apt-get update && \
    apt-get install -y docker.io && \
    apt-get clean

# Switch back to airflow user
USER airflow

# Install Docker SDK for Python
RUN pip install docker



# # Use the official Airflow image as the base image
# FROM apache/airflow:2.8.1

# # Set environment variables
# ENV AIRFLOW_HOME=/opt/airflow

# # Copy the requirements file and install any additional dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirementsDockerImage.txt

# # Copy your DAGs, plugins, and other necessary files
# COPY dags/ $AIRFLOW_HOME/dags/
# COPY plugins/ $AIRFLOW_HOME/plugins/
# COPY models/ /app/models/

# # Set the entrypoint to the Airflow entrypoint script
# ENTRYPOINT ["entrypoint.sh"]

# # Default command to run the webserver
# CMD ["webserver"]