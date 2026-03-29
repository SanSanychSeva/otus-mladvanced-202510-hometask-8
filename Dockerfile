FROM python:3.13.12-slim-trixie

WORKDIR /app

# installing uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

# installing dependencies by uv
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
RUN uv sync

ENV PYTHONPATH="/app"
ENV PATH="/app/.venv/bin:$PATH"

# copying source code and model with fitted preproc objects
COPY src src
COPY models models
COPY pipelines_scripts pipelines_scripts

# REST API settings
ENV FLASK_APP=pipelines_scripts/ml_maas_rest_api.py
EXPOSE 5000

# running REST API after container starts
CMD ["flask", "run", "--host=0.0.0.0"]