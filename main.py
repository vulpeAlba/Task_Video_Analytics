from src.app.app import App

if __name__ == '__main__':
    # ip_address = "192.168.217.103"
    # url = f"http://{ip_address}/axis-cgi/mjpg/video.cgi"

    # video_src = "pers.mp4"
    video_src = "highway.mp4"
    # video_src = "cars2.mp4"
    App(video_src).run()
