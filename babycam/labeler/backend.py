import logging
from time import strftime
import traceback
from video_sample_manager import VideoSampleManager
from flask import Flask, abort, send_file, request


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("api")

app = Flask(__name__)
sample_manager = VideoSampleManager()


@app.route("/")
def ui():
    return "Hello"


@app.get("/samples")
def samples():
    sample_manager.read_samples()
    return sample_manager.get_samples()


@app.get("/samples/<sample_id>")
def sample(sample_id: str):
    return send_file(sample_manager.get_sample(sample_id)["path"])


@app.patch("/samples/<sample_id>")
def patch_sample(sample_id: str):
    sample_manager.set_label(sample_id, request.get_json()["label"])
    sample_manager.set_ready(sample_id)
    return dict()


@app.after_request
def after_request(response):
    timestamp = strftime("[%Y-%b-%d %H:%M]")
    logger.info(
        "%s %s %s %s %s %s",
        timestamp,
        request.remote_addr,
        request.method,
        request.scheme,
        request.full_path,
        response.status,
    )
    return response


@app.errorhandler(Exception)
def exceptions(e):
    tb = traceback.format_exc()
    timestamp = strftime("[%Y-%b-%d %H:%M]")
    logger.error(
        "%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s",
        timestamp,
        request.remote_addr,
        request.method,
        request.scheme,
        request.full_path,
        tb,
    )
    return e


if __name__ == "__main__":
    app.run(port=4000, debug=False)
