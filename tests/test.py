import os
import ssl
import argparse
import base64
from logger import get_logger

import asyncio
import aiohttp


logger = get_logger()
sslcontext = False

allowed_extensions = ['.jpg', '.png']


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


async def compare_face_test(faces_directory: str, base_url: str, headers=None):
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
                    print("success:", response_json["success"])

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
    except aiohttp.ClientError:
        logger.exception('Exception on compare_face_test')


async def get_token(user_id: str, base_url: str):
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
        logger.exception('Exception on get_token')

    return token


parser = argparse.ArgumentParser()
parser.add_argument("--detect_faces_path", type=str, required=True,
                    help="directory with images to test detection")
parser.add_argument("--compare_faces_path", type=str, required=True,
                    help="directory with images to test verification")
parser.add_argument("--facerec_service_ip", type=str,
                    help="ip-address of a docker container with facerec service",
                    default="127.0.0.1")
args = parser.parse_args()

if __name__ == '__main__':
    base_url = f"http://{args.facerec_service_ip}:80/api/v1"

    loop = asyncio.get_event_loop()
    token = loop.run_until_complete(
        asyncio.ensure_future(get_token("test_user", base_url)))

    print('Access token:', token)

    headers = {
        'token': token,
    }

    tasks = [
        encode_face_test(args.detect_faces_path, base_url, headers),
        compare_face_test(args.compare_faces_path, base_url, headers),
    ]
    futures = (asyncio.ensure_future(task) for task in tasks)

    loop.run_until_complete(asyncio.gather(*futures))
    loop.close()
