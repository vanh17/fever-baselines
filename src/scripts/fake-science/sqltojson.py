import mysql.connector
import json
import sys

mydb = mysql.connector.connect(
  host="localhost",
  user="henry",
  passwd="grannygenna",
  database="fakenews_agv"
)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        exit("[usage]: python3 sqltojson.py [climate|astronomy|evolution|physics|...] path_to_output_folder")
    cur = mydb.cursor()
    cur.execute("SELECT sourceID, student, url, category, reality, cesspool, keywords, title, body, taggedBy, manuallySkipped FROM sources WHERE category='"+ sys.argv[1] + "'")
    row_headers = [x[0] for x in cur.description]  # this will extract row headers

    result = cur.fetchall()
    json_data = []

    f = open(sys.argv[2] + "/" + sys.argv[1] + "-corpus.jsonl","w+")
    for x in result:
        f.write(json.dumps(dict(zip(row_headers, x))) + '\n')

    f.close()
