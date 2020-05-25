from flask import Flask, Response

from cv import gen_video

app = Flask(__name__)


@app.route('/camera')
def get_video():
    try:
        return Response(gen_video(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        print(e)
        return "Internal server error", 500


if __name__ == '__main__':
    app.run()
