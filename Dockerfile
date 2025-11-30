# Używamy lekkiej wersji Pythona 3.12
FROM python:3.12-slim

# Ustawiamy katalog roboczy wewnątrz kontenera
WORKDIR /app

# Instalujemy narzędzia systemowe (curl jest przydatny do testów healthcheck)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Kopiujemy plik z zależnosciami
COPY requirements.txt .

# Instalujemy biblioteki Python
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy cały kod źródłowy do kontenera
COPY src/ ./src/

# Ustawiamy zmienną środowiskową, żeby Python widział moduł 'src'
ENV PYTHONPATH=/app

# Domyślna komenda (zostanie nadpisana w docker-compose)
CMD ["python"]
