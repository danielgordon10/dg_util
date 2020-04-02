import glob
import http
import itertools
import json
import os
import re
import socket
import time
import traceback
import urllib
from typing import Optional

import youtube_dl
import youtube_dl.extractor.youtube as yt_extractor
from dg_util.python_utils import misc_util
from youtube_dl.compat import compat_urllib_parse
from youtube_dl.utils import orderedSet

DEBUG = False


def get_video_url(video_id: str) -> str:
    url = "https://www.youtube.com/watch?v=%s" % video_id
    return url


def ydl_download(video_id, ydl_opts):
    """
    Downloads from YDL but with error handling and shit
    :param ydl_opts:
    :return: True if success!
    """
    with misc_util.timeout(60):
        try:
            ydl = youtube_dl.YoutubeDL(ydl_opts)
            result = ydl.download([video_id])
            if result != 0:
                print("Could not download", get_video_url(video_id))
                return False
            return True
        except youtube_dl.DownloadError as e:
            if "Too Many Requests" in str(e):
                # There is no recovering from a toomanyrequests error
                raise e
                # print("sleeping bc toomanyrequests", flush=True)
                # time.sleep(random.randint(1+2*i, 5+5*i))
            else:
                print("Oh no! Problem \n\n{}\n".format(str(e)))
                return False
        except youtube_dl.utils.ExtractorError:
            print("The video is private")
            return False
        except (socket.gaierror, urllib.error.URLError, urllib.error.HTTPError):
            # print("Socket error, retry", video_id)
            if DEBUG:
                print("Socket error, retry", video_id)
        except http.client.RemoteDisconnected:
            print("remote disconnect, retry", video_id)
            if DEBUG:
                print("remote disconnect, retry", video_id)
        except (json.decoder.JSONDecodeError, TypeError):
            print("JSON parsing error", video_id)
            with open("experiment_scripts/failed_files/json_%s.txt" % video_id, "w") as ff:
                ff.write("%s %s\n" % (video_id, video_id))
                traceback.print_exc(file=ff)
                ff.write("JSONError")
            if DEBUG:
                print("JSON parsing error", video_id)
            return False
        except TimeoutError:
            print("Timeout error", video_id)
            return False
        except Exception as e:
            traceback.print_exc()
            with open("experiment_scripts/failed_files/download_recursion_depth_%s.txt" % video_id, "w") as ff:
                ff.write("%s \n" % video_id)
                ff.write("Recursion depth reached")
            print("Some problem with video", video_id)
            print("Misc exception: {}".format(str(e)))
            return False


def download_video_ytdl_helper(id, cache_path, cookie_path=None):
    """
    Given an ID, download the video using HQ settings. (might need to turn this down, idk)

    :param id: Video id
    :param cache_path: Where to download the video
    :return: The file that we downloaded things to
    """

    ydl_opts = {
        "quiet": True,
        "cachedir": cache_path,
        "format": "best[height<=480][ext=mp4]",
        "outtmpl": os.path.join(cache_path, "%(id)s.%(ext)s"),
        "retries": 30,
        "ignoreerrors": True,
        # "source_address": "0.0.0.0",
        "socket_timeout": 30,
        "youtube_include_dash_manifest": False,
        "cookiefile": cookie_path,
        "force-ipv4": True,
    }
    if ydl_download(id, ydl_opts):
        return os.path.join(cache_path, f"{id}.mp4")
    return None


def download_video_ytdl(
    video_id: str, video_path: str = "data/videos", cookie_path: Optional[str] = None
) -> Optional[str]:
    if not os.path.exists(video_path):
        os.makedirs(video_path, exist_ok=True)

    potential_file = glob.glob(os.path.join(video_path, video_id + ".mp4"))
    if len(potential_file) > 0:
        if os.stat(potential_file[0]).st_size == 0:
            return None
        return potential_file[0]

    t_start = time.time()
    if DEBUG:
        print("downloading")
    video = download_video_ytdl_helper(video_id, video_path, cookie_path)
    if video is None or not os.path.exists(video):
        print("download error for ", video_id)
        return None
    if os.stat(video).st_size == 0:
        print("Download failed for", video)
        if os.path.exists(video):
            os.remove(video)
        return None
    if DEBUG:
        print("download done in %.3f" % (time.time() - t_start))
    return video


def download_video(video_id: str, video_path: str = "data/videos", cookie_path: Optional[str] = None) -> Optional[str]:
    return download_video_ytdl(video_id, video_path, cookie_path)


class YoutubeSearchIECC(yt_extractor.YoutubeSearchIE):
    def _get_n_results(self, query, n, search_str=""):
        """Get a specified number of results for a query"""

        videos = []
        channels = set()
        limit = n

        for pagenum in itertools.count(1):
            url_query = {"search_query": query.encode("utf-8"), "page": pagenum, "spf": "navigate"}
            url_query.update(self._EXTRA_QUERY_ARGS)
            if len(search_str) == 0:
                search_str = "CAISBhABGAEwAQ%253D%253D"  # Video, Short (<4 minutes), Creative Commons, Upload Date
            result_url = (
                "https://www.youtube.com/results?sp=" + search_str + "&" + compat_urllib_parse.urlencode(url_query)
            )
            data = self._download_json(
                result_url,
                video_id='query "%s"' % query,
                note="Downloading page %s" % pagenum,
                errnote="Unable to download API page",
            )
            html_content = data[1]["body"]["content"]

            if 'class="search-message' in html_content:
                raise Exception("[youtube] No video results", result_url)

            # video_urls = re.findall(r'href="/watch\?v=(.{11})', html_content)

            vids_and_channels = re.findall(r'href="\/watch\?v=(.{11}).+(?:user|channel)\/(.+?")', html_content)
            # print('got vids', len(videos), 'for', query)
            if len(vids_and_channels) == 0:
                break

            for vid, channel in vids_and_channels:
                if channel in channels:
                    continue
                channels.add(channel)
                videos.append(vid)

            # videos += new_videos
            if len(videos) > limit:
                break

        if len(videos) > n:
            videos = videos[:n]
        videos = self._ids_to_results(orderedSet(videos))
        return self.playlist_result(videos, query)


def search_youtube(query, n, search_str=""):
    ydl_opts = {
        "quiet": True,
        "format": "best[height<=480][ext=mp4]",
        "retries": 20,
        "ignoreerrors": True,
        # "source_address": "0.0.0.0",
        "socket_timeout": 30,
        "youtube_include_dash_manifest": False,
        "skip_download": True,
        "writeinfojson": False,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        searcher = YoutubeSearchIECC(ydl)
        # Get results
        results = searcher._get_n_results(query, n, search_str)
        entries = results["entries"]
        ids = [entry["id"] for entry in entries]
        return ids
