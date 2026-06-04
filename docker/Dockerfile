FROM python:3.12

# Set working directory
WORKDIR /app

# Install uv for package dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# copy files
COPY . .

# update path to use installed dependencies
ENV PYTHONPATH=/app

# expose api output
EXPOSE 8000

# start code
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# Code to build docker: docker build -f docker/Dockerfile -t eurosat-api .
# names Docker image eurosat-api and uses current context for build context

# code to run continer: docker run -p 8000:8000 eurosat-api

# to check if docker is running, check if opening http://localhost:8000 works


