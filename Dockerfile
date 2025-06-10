FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir uv \
    && uv pip install -r requirements.txt
CMD ["bash"]
