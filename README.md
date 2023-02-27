# FaceFastAPI

Simple FastAPI based service for faces comparison


## Build docker image and run container

```
docker build -t facerec:latest .
docker run -d -p 80:80 -v $PWD/app:/facerec/app facerec:latest
```

## Documentation
FastAPI auto-generated docs should be available at http://127.0.0.1/docs after docker container started

## Run tests

```
sudo apt install python3.8-venv

cd ./tests
python3 -m venv .venv
source .venv/bin/activate

pip install aiohttp==3.8.4

python ./test.py --detect_faces_path ./faces/ --compare_faces_path ./matched_faces/
python ./test.py --detect_faces_path ./faces/ --compare_faces_path ./non_matched_faces/
```
