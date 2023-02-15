FROM pypy:latest
WORKDIR /app

# Upgrade pip3
RUN pip3 install --upgrade pip

# install setuptools
RUN pip3 install --upgrade setuptools

# Install spacy
RUN pip3 install spacy==3.4.0

# Install spacy model sm
RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.4.0/en_core_web_md-3.4.0.tar.gz --user

# Copy requirements.txt from computer to image with the same filename
COPY requirements.txt requirements.txt

# Install requirements specified in requirements.txt
RUN pip3 install -r requirements.txt

# Copy all files from directory where requirements.txt is located.
COPY . .

CMD python semantic.py