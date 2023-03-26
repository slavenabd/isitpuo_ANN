# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory to /app
WORKDIR /

# Copy the requirements file into the container at /app
COPY requirements.txt /

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /

# Set the default command to run when the container starts
CMD ["python", "isitpuo_webpage_flask.py"]
