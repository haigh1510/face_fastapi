# FaceFastAPI

Simple FastAPI based service for faces comparison


## Build docker image and run container

```
docker-compose -f ./docker-compose.yml up --build -d
```

## Documentation
FastAPI auto-generated docs should be available at http://127.0.0.1/docs after docker container started

## Run tests

```
sudo apt install python3.8-venv

python3 -m venv .venv
source .venv/bin/activate

pip install -r ./tests/requirements.txt
pytest -s ./tests
```
