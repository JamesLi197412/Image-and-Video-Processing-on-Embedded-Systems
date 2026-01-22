# Example (adjust duration, fps, resolution):
#rpicam-vid -t 10000 --framerate 10 --width 640 --height 360 -o video.h264
#ffmpeg -y -i video.h264 -c:v libx264 -movflags faststart video.mp4
