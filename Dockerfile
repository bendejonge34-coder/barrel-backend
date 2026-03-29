FROM node:20-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm install

COPY requirements.txt ./

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV NODE_ENV=production
ENV PYTHON_PATH=/opt/venv/bin/python

CMD ["node", "server.js"]