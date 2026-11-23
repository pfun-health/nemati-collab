# Minimal image for running pycardio. The only build requirement is a C compiler.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml setup.py README.md ./
COPY pycardio ./pycardio
RUN pip install --no-cache-dir .

# Mount the folder that contains fantasia/ here, e.g. -v ~/Public:/data
ENV PYCARDIO_DATA_DIR=/data
ENTRYPOINT ["pycardio"]
CMD ["--help"]
