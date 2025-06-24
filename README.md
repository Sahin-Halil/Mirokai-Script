# Mirokai Script

A Python script built during **Robohack 2025** to enable the Mirokai robot to perceive its surroundings, describe them, and speak fun facts aloud, all in real time.

This project showcases integration of computer vision, natural language processing, and robotic speech using a combination of OpenCV, BLIP, spaCy, and Wikipedia API.

---

## Features

- Captures real-time images from Mirokai's RTSP camera feed using OpenCV
- Uses the BLIP image captioning model to describe the image
- Extracts nouns from the caption using spaCy
- Queries the Wikipedia API for fun facts based on detected nouns
- Speaks the result using Mirokai's built-in `robot.say()` method
- Testing was done with a simulation of Mirokai on the Gazebo Simulation

---

## Demo

 **Watch the robot in action:** [https://youtu.be/Rsk4kfzCyzI]  

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Sahin-Halil/Mirokai-Script.git
   cd Mirokai-Script
   ```

2. Install Python dependencies:
   - OpenCV:
     ```bash
     pip install opencv-python
     ```
     
   - Transformers (for BLIP)
     ```bash
     pip install transformers
     ```
     
   - PyTorch (required for BLIP to run):
     ```bash
     pip install torch
     ```
     
   - spaCy
     ```bash
     pip install spacy
     ```
     
3. To run the script you have two options:

    **Option A — Run on the Physical Mirokai Robot**

   - Make sure the Mirokai robot is powered on and connected to the same network.
   - Identify the robot's IP address and API key.
   - Run the script:
     ```bash
     python Mirokai_Script.py --ip <ROBOT_IP> --api-key <API_KEY>
     ```

   **Option B — Run with Gazebo Simulation**

    - If you don't have access to a physical Mirokai robot, you can test the script in a simulated environment using **Gazebo**.
    - Make sure Docker and Docker Compose are installed.
    - Follow the official documentation to set up the Mirokai Gazebo simulation:  
      ```bash
      https://gazebosim.org/docs/latest/getstarted/
      ```
    
    - Once your simulation is running, use the appropriate simulated IP and run:
      ```bash
      python Mirokai_Script.py --ip <SIMULATED_IP> --api-key <API_KEY>
      ```
    - **Note:** The Gazebo simulation replicates image capture and processing but does not support audio output, so you won’t hear the robot speak.

## Tech Stack

- **Image Capture**: OpenCV, RTSP
- **Captioning**: BLIP (Hugging Face Transformers)
- **Natural Language Processing**: spaCy (`en_core_web_sm`)
- **Fun Fact Retrieval**: Wikipedia REST API
- **Speech Output**: `robot.say()` (pymirokai SDK)
- **Simulation Environment**: Gazebo, Docker Compose

## Notes
- This project was developed during a 2-day hackathon under time constraints.
- Due to limited documentation, we reverse-engineered functionality from the pymirokai SDK.
- The entire pipeline was tested in a Gazebo simulation before being run live on the robot for the first time during the final demo.

## Credits:
Built by **Sahin Halil** and **Rayyan Parkar** at **Robohack 2025**.



     
   
