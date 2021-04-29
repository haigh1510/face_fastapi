import os
import argparse
import base64
from logger import get_logger

import asyncio
import aiohttp


logger = get_logger()

base_url = "http://127.0.0.1:9876/api/v1"

allowed_extensions = ['.jpg', '.png']


async def encode_face_test(faces_directory: str):
    try:
        image_paths = [os.path.join(faces_directory, f) for f in os.listdir(faces_directory) if any([f.endswith(ext) for ext in allowed_extensions])]

        async with aiohttp.ClientSession() as session:
            for image_path in image_paths:
                print('Processing image:', image_path)

                with open(image_path, "rb") as image:
                    image_data = image.read()

                image_base64 = base64.b64encode(image_data)
                json_content = {"image": image_base64.decode('utf-8')}

                async with session.request('POST', base_url + "/encode_face", json=json_content) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    print('response:', response_json)
    except aiohttp.ClientError:
        logger.exception('Exception on encode_face_test')


async def compare_face_test(faces_directory: str):
    try:
        data_to_verify = []
        image_paths = [os.path.join(faces_directory, f) for f in os.listdir(faces_directory) if any([f.endswith(ext) for ext in allowed_extensions])]

        assert len(image_paths) == 2, f"Exactly 2 images expected to compare, found {len(image_paths)}"

        async with aiohttp.ClientSession() as session:
            for image_path in image_paths:
                print('Processing image:', image_path)

                with open(image_path, "rb") as image:
                    image_data = image.read()

                image_base64 = base64.b64encode(image_data)
                request = {"image": image_base64.decode('utf-8')}

                async with session.request('POST', base_url + "/encode_face", json=request) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    print("success:", response_json["success"])

                    if response_json["success"]:
                        data_to_verify.append((request["image"], response_json["face_encoding"]))

            assert len(data_to_verify) == 2

            db_face_data = data_to_verify[0]
            verify_face_data = data_to_verify[1]

            compare_request = {
                "db_face_encoding": db_face_data[1],
                "image": verify_face_data[0]
            }

            async with session.request('POST', base_url + "/compare_faces", json=compare_request) as response:
                response.raise_for_status()
                response_json = await response.json()
                print("verify_distance:", response_json["verify_distance"])
    except aiohttp.ClientError:
        logger.exception('Exception on compare_face_test')


parser = argparse.ArgumentParser()
parser.add_argument("--detect_faces_path", type=str, required=True,
                    help="directory with images to test detection")
parser.add_argument("--compare_faces_path", type=str, required=True,
                    help="directory with images to test verification")
args = parser.parse_args()

if __name__ == '__main__':
    tasks = [
        encode_face_test(args.detect_faces_path),
        compare_face_test(args.compare_faces_path),
    ]
    futures = (asyncio.ensure_future(task) for task in tasks)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*futures))
    loop.close()
