import subprocess
import json
import random
import time
import os

subprocess.run(["make", "vanity-board"])

while True:
    r = hex(random.randint(0, 255))
    g = hex(random.randint(0, 255))
    b = hex(random.randint(0, 255))
    if not os.path.exists("job.json"):
        with open("job.json", "w") as f:
            result = subprocess.run(["painter", "job", "get"], stdout=f)
            print(f"Got job: {result}")
            if result.returncode != 0:
                print("No job available", result.stderr, time.time())
                time.sleep(1)
                continue
        with open("job.json") as f:
            job = json.load(f)
            print(f"Got job: {job}")
    else:
        with open("job.json") as f:
            job = json.load(f)
            print(f"Got job: {job}")

    R = r + job["r"]
    G = g + job["g"]
    B = b + job["b"]

    with open("vanity.in", "w") as f:
        f.write(f"{R}\n{G}\n{B}")

    vanity = subprocess.run(["./vanity-board", R, G, B], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    print(f"Vanity board: {vanity.stdout}")

    with open("vanity.out", "w") as f:
        f.write(vanity.stdout)

    vanity.check_returncode()
    results = vanity.stdout.split()
    RKey, GKey, BKey = results[1], results[3], results[5]
    print(f"RKey: {RKey}, GKey: {GKey}, BKey: {BKey}")

    token_json = subprocess.run(["painter", "job", "submit", "--r", RKey, "--g", GKey, "--b", BKey, "--jobid", job["jobid"]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    print(' '.join(token_json.args))
    print(f"Submitted job: {token_json}")
    token_json.check_returncode()
    token = json.loads(token_json.stdout)["token"]

    i = 0
    while i < 10:
        x = random.randint(0, 800)
        y = random.randint(0, 600)
        while 255 <= x <= 320 and 345 <= y <= 480:
            x = random.randint(0, 800)
            y = random.randint(0, 600)
        subprocess.run(["pointer", "pixel", "set", "--x", x, "--y", y, "--token", token], stdout=subprocess.STDOUT, stderr=subprocess.STDOUT, encoding="utf-8")
        i += 1
        os.rename("job.json", f"job.json.done.{time.time()}")
