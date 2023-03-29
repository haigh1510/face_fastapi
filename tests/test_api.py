import pytest

import os
import ssl
import base64
import asyncio
import aiohttp

from .logger import get_logger


logger = get_logger()
sslcontext = False

allowed_extensions = ['.jpg', '.png']


@pytest.mark.skip()
async def encode_face_test(faces_directory: str, base_url: str, headers=None):
    try:
        image_paths = [os.path.join(faces_directory, f) for f in os.listdir(faces_directory) if any([f.endswith(ext) for ext in allowed_extensions])]

        async with aiohttp.ClientSession(headers=headers) as session:
            for image_path in image_paths:
                print('Processing image:', image_path)

                with open(image_path, "rb") as image:
                    image_data = image.read()

                image_base64 = base64.b64encode(image_data)
                json_content = {"image": image_base64.decode('utf-8')}

                async with session.request('POST', base_url + "/encode_face",
                                           json=json_content,
                                           ssl=sslcontext) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    print('response:', response_json)
    except aiohttp.ClientError:
        logger.exception('Exception on encode_face_test')


@pytest.mark.skip()
async def _compare_face(faces_directory: str, base_url: str, headers=None):
    verified = False

    try:
        data_to_verify = []
        image_paths = [os.path.join(faces_directory, f) for f in os.listdir(faces_directory) if any([f.endswith(ext) for ext in allowed_extensions])]

        assert len(image_paths) == 2, f"Exactly 2 images expected to compare, found {len(image_paths)}"

        async with aiohttp.ClientSession(headers=headers) as session:
            for image_path in image_paths:
                print('Processing image:', image_path)

                with open(image_path, "rb") as image:
                    image_data = image.read()

                image_base64 = base64.b64encode(image_data)
                request = {"image": image_base64.decode('utf-8')}

                async with session.request('POST', base_url + "/encode_face",
                                           json=request,
                                           ssl=sslcontext) as response:
                    response.raise_for_status()
                    response_json = await response.json()

                    print("response_json:", response_json)

                    if response_json["success"]:
                        data_to_verify.append(
                            {
                                "image": request["image"],
                                "face_encoding": response_json["face_encoding"]
                            }
                        )

            assert len(data_to_verify) == 2, f"Exactly 2 elements expected (gallery and query), found {len(data_to_verify)}"

            gallery, query = data_to_verify

            compare_request = {
                "db_face_encoding": gallery["face_encoding"],  # gallery encoded face
                "image": query["image"]                        # query image
            }

            async with session.request('POST', base_url + "/compare_faces",
                                       json=compare_request,
                                       ssl=sslcontext) as response:
                response.raise_for_status()
                response_json = await response.json()
                if not response_json["success"]:
                    print('message:', response_json["message"])
                print("verification:", response_json["verification"])
                print("seconds:", response_json["seconds"])

                verified = response_json["verification"]["verified"]
    except aiohttp.ClientError:
        logger.exception('Exception on compare_face_test')

    return verified


@pytest.mark.skip()
async def compare_face_match(faces_directory: str, base_url: str, headers=None):
    verified = await _compare_face(faces_directory, base_url, headers)
    assert verified


@pytest.mark.skip()
async def compare_face_mismatch(faces_directory: str, base_url: str, headers=None):
    verified = await _compare_face(faces_directory, base_url, headers)
    assert not verified


@pytest.fixture
def get_token():
    async def token_request(user_id: str, base_url: str):
        token = None
        try:
            async with aiohttp.ClientSession() as session:
                json_content = {"user_id": user_id}

                async with session.request('POST', base_url + "/token",
                                        json=json_content,
                                        ssl=sslcontext) as response:
                    response.raise_for_status()
                    response = await response.json()

                if response["success"]:
                    token = response["token"]["access_token"]
        except aiohttp.ClientError:
            logger.exception('Exception on token_request')

        return token

    return token_request


@pytest.mark.asyncio
async def test_api(get_token):
    base_url = f"http://127.0.0.1:80/api/v1"

    token = await asyncio.ensure_future(get_token("test_user", base_url))
    print('Access token:', token)

    assert token is not None

    headers = {
        'token': token,
    }

    basepath = os.path.dirname(__file__)

    tasks = [
        encode_face_test(os.path.join(basepath, "./faces/"), base_url, headers),
        compare_face_match(os.path.join(basepath, "./matched_faces/"), base_url, headers),
        compare_face_mismatch(os.path.join(basepath, "./non_matched_faces/"), base_url, headers)
    ]
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
        except Exception as e:
            raise e
