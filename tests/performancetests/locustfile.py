import os
import time
from locust import HttpUser, task, between
from locust.contrib.fasthttp import FastHttpUser
from locust import events
from io import BytesIO
from PIL import Image

class RiceClassificationUser(FastHttpUser):
    wait_time = between(1, 2) 

    @task
    def upload_image(self):
        image_path = "tests/sample_image.jpg"
        with open(image_path, "rb") as f:
            file_data = f.read()

        with self.client.post(
            "/predict/",
            files={"data": ("sample_image.jpg", BytesIO(file_data), "image/jpg")},
            catch_response=True,
        ) as response:

            if response.status_code == 200:
                print(f"Prediction successful: {response.json()}")
            else:
                response.failure(f"Prediction failed with status code {response.status_code}")



"""
Locust report

>> locust -f tests/performancetests/locustfile.py --headless --users 10 --spawn-rate 1 --run-time 30s --host http://localhost:8089
[2025-01-24 12:06:15,830] DESKTOP-UHS6K67/INFO/locust.main: Starting Locust 2.32.6
[2025-01-24 12:06:15,832] DESKTOP-UHS6K67/INFO/locust.main: Run time limit set to 30 seconds
Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated       0     0(0.00%) |      0       0       0      0 |    0.00        0.00

[2025-01-24 12:06:15,832] DESKTOP-UHS6K67/INFO/locust.runners: Ramping to 10 users at a rate of 1.00 per second
Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated       0     0(0.00%) |      0       0       0      0 |    0.00        0.00

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated       0     0(0.00%) |      0       0       0      0 |    0.00        0.00

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/       2   2(100.00%) |   4061    4051    4071   4071 |    0.00        0.00
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated       2   2(100.00%) |   4061    4051    4071   4071 |    0.00        0.00

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/       4   4(100.00%) |   4053    4042    4071   4042 |    0.20        0.20
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated       4   4(100.00%) |   4053    4042    4071   4042 |    0.20        0.20

[2025-01-24 12:06:24,880] DESKTOP-UHS6K67/INFO/locust.runners: All users spawned: {"RiceClassificationUser": 10} (10 total users)
Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/       7   7(100.00%) |   4048    4034    4071   4034 |    0.50        0.50
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated       7   7(100.00%) |   4048    4034    4071   4034 |    0.50        0.50

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/      11  11(100.00%) |   4050    4034    4071   4071 |    0.60        0.60
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated      11  11(100.00%) |   4050    4034    4071   4071 |    0.60        0.60

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/      15  15(100.00%) |   4056    4034    4091   4091 |    1.00        1.00
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated      15  15(100.00%) |   4056    4034    4091   4091 |    1.00        1.00

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/      18  18(100.00%) |   4059    4034    4091   4091 |    1.40        1.40
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated      18  18(100.00%) |   4059    4034    4091   4091 |    1.40        1.40

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/      23  23(100.00%) |   4060    4031    4091   4091 |    1.40        1.40
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated      23  23(100.00%) |   4060    4031    4091   4091 |    1.40        1.40

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/      25  25(100.00%) |   4058    4031    4091   4091 |    1.60        1.60
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated      25  25(100.00%) |   4058    4031    4091   4091 |    1.60        1.60

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/      29  29(100.00%) |   4058    4031    4091   4091 |    1.80        1.80
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated      29  29(100.00%) |   4058    4031    4091   4091 |    1.80        1.80

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/      33  33(100.00%) |   4058    4031    4091   4091 |    1.80        1.80
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated      33  33(100.00%) |   4058    4031    4091   4091 |    1.80        1.80

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/      37  37(100.00%) |   4059    4031    4091   4091 |    1.80        1.80
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated      37  37(100.00%) |   4059    4031    4091   4091 |    1.80        1.80

Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/      40  40(100.00%) |   4058    4031    4091   4091 |    1.90        1.90
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated      40  40(100.00%) |   4058    4031    4091   4091 |    1.90        1.90

[2025-01-24 12:06:45,305] DESKTOP-UHS6K67/INFO/locust.main: --run-time limit reached, shutting down
[2025-01-24 12:06:45,491] DESKTOP-UHS6K67/INFO/locust.main: Shutting down (exit code 1)
Type     Name  # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /predict/      43  43(100.00%) |   4056    4024    4091   4091 |    1.48        1.48
--------||-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated      43  43(100.00%) |   4056    4024    4091   4091 |    1.48        1.48

Response time percentiles (approximated)
Type     Name      50%    66%    75%    80%    90%    95%    98%    99%  99.9% 99.99%   100% # reqs
--------||--------|------|------|------|------|------|------|------|------|------|------|------
POST     /predict/     4100   4100   4100   4100   4100   4100   4100   4100   4100   4100   4100     43
--------||--------|------|------|------|------|------|------|------|------|------|------|------
         Aggregated     4100   4100   4100   4100   4100   4100   4100   4100   4100   4100   4100     43


Error report
# occurrences      Error                                             

------------------|-------------------------------------------------------------
43                 POST /predict/: Prediction failed with status code 0
------------------|-------------------------------------------------------------


"""
