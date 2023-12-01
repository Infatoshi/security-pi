import cv2
import numpy as np
import discord
import asyncio
from discord.ext import commands
import time
from discord import File
import datetime
from dotenv import load_dotenv
import os

load_dotenv()

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")

CHANNEL_ID = int(os.environ.get("CHANNEL_ID"))

USER_ID = int(os.environ.get("USER_ID"))

filter_time = 0
delay = 3

intents = discord.Intents.all()
bot = discord.Client(intents=intents)

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)
def is_motion_detected(video_source=0, threshold=1000, max_frames=5):
    
    desired_width = 640
    desired_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    frames = []
    ret, frame1 = cap.read()
    ret, frame05 = cap.read()
    ret, frame2 = cap.read()
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > threshold:
                frames.append(frame2)
                if len(frames) > max_frames:
                    frames.pop(0)
                
                cv2.imwrite(f'images/frame.jpg', frame2)
                return True

        frame1 = frame2
        ret, frame2 = cap.read()

        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

async def check_for_motion():
    while True:
        motion_detected = is_motion_detected()
        if motion_detected:
            # print("Motion detected!")
            await send_motion_detected()
        await asyncio.sleep(1)  # Adjust the sleep duration as needed

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    bot.loop.create_task(check_for_motion())

async def send_motion_detected():
    global filter_time
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        current_time_seconds = time.time()
        if current_time_seconds - filter_time > delay:
            filter_time = current_time_seconds
            current_time = datetime.datetime.fromtimestamp(current_time_seconds)
            formatted_date = current_time.strftime("%Y-%m-%d %H:%M:%S")
            message_content = f'*<@{USER_ID}> Motion Detected at {formatted_date}*'

            # Path to your image
            image_path = 'images/frame.jpg'  # Update this path as needed

            # Create a File object
            image_file = File(image_path)

            # Send the message with the image
            await channel.send(message_content, file=image_file)
    else:
        print("Channel not found.")


async def main():
    await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
