# INT-Vision
Crowd analysis tool

INT-Vision (Interest Vision) is a desktop software to analyse productivity of multiple human subjects simultaneously based on emotions and gestures. It can be easily deployed on any device with a digital camera and both real-time and graph analysis results can be seen on the desktop with the click of a button.  
  
Possible applications:  
- In schools/coachings to analyse interest of students in a particular lecture at various points of time.  
- In offices to analyse productivity of employees.  
- In workshops/seminars to analyse the interest of audience in the presented content.  

This serves as an AI enhanced surveillance system and eliminates the need for feedback or human monitoring.  

How to use:
1. Navigate to the algo folder.
```shell
cd algo
```
2. Run main file using Python.
```shell
python3 main.py
```
3. Press "Start Webcam" to start analysis using default laptop/PC camera. Press Esc key to stop recording. Wait for processing to finish.
4. Press "Start Video" to see analysed video with blue boxes indicating interested subjected and red boxes indicating disengaged, sleepy or bored subjects.
5. Press "Show Graph" to see a graph of interest level of people versus time.
