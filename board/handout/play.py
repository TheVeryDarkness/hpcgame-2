import subprocess
import json
import random

subprocess.run(["make", "vanity-board"])

while True:
    result = subprocess.run(["painter", "job", "get"])
    result.check_returncode()
    job = json.loads(result.stdout)
    print(f"Got job: {job}")

    vanity = subprocess.run(["./vanity-board", job["r"], job["g"], job["b"]], encoding="utf-8")
    vanity.check_returncode()
    results = vanity.stdout.decode().split()
    RKey, GKey, BKey = results[0], results[1], results[2]
    print(f"RKey: {RKey}, GKey: {GKey}, BKey: {BKey}")

    token_json = subprocess.run(["painter", "job", "submmit", "--r", RKey, "--g", GKey, "--b", BKey])
    token_json.check_returncode()
    token = json.loads(token_json.stdout)["token"]

    i = 0
    while i < 10:
        x = random.randint(0, 800)
        y = random.randint(0, 600)
        while 255 <= x <= 320 and 345 <= y <= 480:
            x = random.randint(0, 800)
            y = random.randint(0, 600)
        subprocess.run(["pointer", "pixel", "set", "--x", x, "--y", y, "--token", token])
        i += 1
