import os
import base64
from logger import get_logger

import asyncio
import aiohttp


logger = get_logger()

base_url = "http://127.0.0.1:9876/api/v1"
faces_directory = "FACES"

same_face_directory = "same_faces"
different_face_directory = "different_faces"

compare_face_directory = same_face_directory
# compare_face_directory = different_face_directory


async def encode_face_test():
    try:
        image_paths = [os.path.join(faces_directory, f) for f in os.listdir(faces_directory) if '.jpg' in f]
        for image_path in image_paths:
            print('Processing image:', image_path)

            with open(image_path, "rb") as image:
                image_data = image.read()

            image_base64 = base64.b64encode(image_data)
            json_content = {"image": image_base64.decode('utf-8')}

            async with aiohttp.request('POST', base_url + "/encode_face", json=json_content) as response:
                response.raise_for_status()
                response_json = await response.json()
                print('response:', response_json)
    except aiohttp.ClientError:
        logger.exception('Exception on encode_face_test')


async def compare_face_test():
    try:
        data_to_verify = []
        image_paths = [os.path.join(compare_face_directory, f) for f in os.listdir(compare_face_directory) if '.jpg' in f]
        for image_path in image_paths:
            print('Processing image:', image_path)

            with open(image_path, "rb") as image:
                image_data = image.read()

            image_base64 = base64.b64encode(image_data)
            request = {"image": image_base64.decode('utf-8')}

            async with aiohttp.request('POST', base_url + "/encode_face", json=request) as response:
                response.raise_for_status()
                response_json = await response.json()
                print("success:", response_json["success"])

                data_to_verify.append((request["image"], response_json["face_encoding"]))

        assert len(data_to_verify) == 2

        db_face_data = data_to_verify[0]
        verify_face_data = data_to_verify[1]

        compare_request = {
            "db_face_encoding": db_face_data[1],
            "image": verify_face_data[0]
        }

        async with aiohttp.request('POST', base_url + "/compare_faces", json=compare_request) as response:
            response.raise_for_status()
            response_json = await response.json()
            print("verify_distance:", response_json["verify_distance"])
    except aiohttp.ClientError:
        logger.exception('Exception on compare_face_test')


if __name__ == '__main__':
    tasks = [
        encode_face_test(),
        compare_face_test(),
    ]
    futures = (asyncio.ensure_future(task) for task in tasks)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*futures))
    loop.close()
