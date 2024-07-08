#!/usr/bin/env python3

# standard library modules
import sys, errno, re, json, ssl
from urllib import request
from urllib.error import HTTPError
from time import sleep

BASE_URL = "https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/pfam/PF00394/?page_size=200"

def output_list():
  #disable SSL verification to avoid config issues
  context = ssl._create_unverified_context()
  print("Downloading Pfam accessions from " + BASE_URL)

  file = open("pfam_accessions.txt", "w")
  next = BASE_URL
  last_page = False

  
  attempts = 0
  while next:
    try:
      req = request.Request(next, headers={"Accept": "application/json"})
      res = request.urlopen(req, context=context)
      # If the API times out due a long running query
      if res.status == 408:
        # wait just over a minute
        sleep(61)
        # then continue this loop with the same URL
        continue
      elif res.status == 204:
        #no data so leave loop
        break
      payload = json.loads(res.read().decode())
      next = payload["next"]
      print(next)
      attempts = 0
      if not next:
        last_page = True
    except HTTPError as e:
      if e.code == 408:
        print("API timed out, waiting 1 seconds and trying again")
        sleep(1)
        continue
      else:
        # If there is a different HTTP error, it wil re-try 3 times before failing
        if attempts < 3:
          attempts += 1
          print("HTTP error " + str(e.code) + ", waiting 1 seconds and trying again")
          sleep(1)
          continue
        else:
          sys.stderr.write("LAST URL: " + next)
          raise e

    for i, item in enumerate(payload["results"]):
      file.write(item["metadata"]["accession"] + "\n")
      file.flush()
    # Don't overload the server, give it time before asking for more
    # if next:
    #   sleep(1)
  file.close()

if __name__ == "__main__":
  output_list()
