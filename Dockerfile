FROM python:3.11-slim

WORKDIR /app

# Copiar requirements.txt primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY agent.py .
# Exponer el puerto que usa la aplicación
EXPOSE 8900

# Comando para ejecutar la aplicación
CMD ["python", "agent.py"]


