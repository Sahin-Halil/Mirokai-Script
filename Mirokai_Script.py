import argparse
import asyncio
import logging
from pymirokai.enums import Hand
from pymirokai.enums.enums import HeadMode
from pymirokai.models.data_models import EnchantedObject, Coordinates
from pymirokai.robot import connect, Robot
from pymirokai.utils.get_local_ip import get_local_ip
from pymirokai.utils.run_until_interruption import run_until_interruption
import asyncio
from pymirokai.robot import connect
from pymirokai.models.data_models import Coordinates
from pymirokai.core.video_api import VideoAPI
import time
import cv2
import numpy as np
import random

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


import spacy
import random
import requests

# Entry point to run the robot interaction asynchronously.
async def run(ip: str, api_key: str) -> None:
    """Run the robot, demonstrating various features."""
    print(f"api = {api_key} ip = {ip}")
    async with connect(api_key, ip) as robot:
        await robot_behavior(robot, ip)
        await asyncio.Future()  # Wait indefinitely

# Defines the core behavior: capture image, describe it, extract a noun, fetch fun facts, and have the robot say them.
async def robot_behavior(robot: Robot, ip: str) -> None:
    """Fun facts about whats in front of Mirokai."""

    # Capture a snapshot from the robot's camera feed.
    logger.info("======= ASKING ROBOT WHAT IT SEES =======")
    image_path = save_rtsp_snapshot(ip)

    # Generate a caption from the captured image.
    img_desc = describe_image(image_path)
    print("Image Description:", img_desc)

    # Load spaCy model and extract nouns from the caption.
    print("about to use spacy")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(img_desc)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    print(nouns)
    
    # Randomly choose one noun and fetch fun facts using Wikipedia API.
    print("Parsing nouns through another llm")
    word = random.choice(nouns)
    fun_facts = get_fun_facts(word)

    # Handle case when no facts are found; otherwise, concatenate and speak them.
    if len(fun_facts) == 1:
        await robot.say(f"Sorry, there is no fun facts about {word}").completed()    
    else:
        answer = ""
        for i in range(len(fun_facts)):
            answer += " "  + fun_facts[i]
        await robot.say(answer).completed() 
    return

# Connects to the robot's RTSP video stream and saves a snapshot frame.
def save_rtsp_snapshot(robot_ip: str, stream_name: str = "head_color"):
    print("here 1")

    # Construct the full RTSP URL for the robot's video stream.
    full_url = f"rtsp://{robot_ip}:8554/{stream_name}"
    print("here 2")

    # Initialize the video stream API with no display window and a timeout.
    video_api = VideoAPI(display=False, timeout=5000)
    video_api.start(full_url)
    time.sleep(2)  
    num = np.random.randint(1000, 9999)

    print("here 3")
    try:
        frame = None

        # Continuously attempt to read a frame from the stream until a valid frame is received.
        while frame is None:
            frame = video_api.get_current_frame()
            if frame is not None:
                # Save the valid frame to a PNG file in the 'frames/' directory.
                filepath = f"frames/{stream_name}_snapshot_{num}.png"
                cv2.imshow(stream_name, frame)
                cv2.imwrite(filepath, frame)
                print(f"Saved {filepath}")
            
            # Exit if 'q' is pressed during the display window.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the video stream, close any OpenCV windows.
        video_api.close()
        cv2.destroyAllWindows()
        return filepath

# Generates a caption for an image using the BLIP image captioning model.
def describe_image(image_path: str) -> str:
     # Load the BLIP processor and model for image captioning.
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Open the image and convert it to RGB format.
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image into model-friendly tensors.
    inputs = processor(image, return_tensors="pt")

    # Generate a caption from the image using the model.
    out = model.generate(**inputs)

    # Decode the output tokens into a readable string
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Fetches a short summary from Wikipedia and extracts up to 3 sentences as fun facts.
def get_fun_facts(noun, num_facts=3):
    # Format the noun for a Wikipedia URL by replacing spaces with dashes.
    title = noun.strip().replace(" ", "-")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    
    # Make a GET request to Wikipedia's summary endpoint.
    response = requests.get(url)

    if response.status_code == 200:
        # If the request is successful, extract and process the summary.
        data = response.json()
        summary = data.get("extract", "")

        # Split the summary into sentences and clean them.
        sentences = summary.split(". ")
        facts = [s.strip() + "." for s in sentences if s.strip()]

        # Return the first few sentences as fun facts.
        return facts[:num_facts]
    
    # If the page does not exist, return a fallback message.
    elif response.status_code == 404:
        return ["No response for '{noun}'."]
    else:
        # Handle other error responses from the API.
        [f"Error: {response.status_code} - Could not fetch data."]

# Parses command-line arguments and launches the main async run function.
async def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-i",
        "--ip",
        help="Set the IP of the robot you want to connect.",
        type=str,
        default=get_local_ip(),
    )
    parser.add_argument(
        "-k",
        "--api-key",
        help="Set the API key of the robot you want to connect.",
        type=str,
        default="",
    )
    args = parser.parse_args()
    await run(ip=args.ip, api_key=args.api_key)

# Starts the async event loop and runs the main function until interruption.
if __name__ == "__main__":
    run_until_interruption(main)
