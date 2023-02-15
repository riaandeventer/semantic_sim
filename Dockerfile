FROM pypy:latest
WORKDIR /app

# Copy requirements.txt from computer to image with the same filename
COPY requirements.txt requirements.txt

# Install requirements specified in requirements.txt
RUN pip3 install -r requirements.txt

# Copy all files from directory where requirements.txt is located.
COPY . .

CMD python symantic.py