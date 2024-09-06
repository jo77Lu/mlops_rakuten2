# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Update pip
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r streamlit_requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

# Run streamlit when the container launches
CMD ["streamlit", "run", "streamlit.py", "--server.port", "8501"]