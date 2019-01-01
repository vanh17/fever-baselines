import mysql.connector
import json

mydb = mysql.connector.connect(
  host="localhost",
  user="yourusername",
  passwd="yourpassword",
  database="mydatabase"
)

if __name__ == "__main__":
    cur = mydb.cursor()
    cur.execute("SELECT * FROM customers")
    row_headers = [x[0] for x in cur.description]  # this will extract row headers

    result = cur.fetchall()
    json_data = []

    f = open("fake-science-corpus.jsonl","w+")
    for x in result:
        f.write(json.dumps(dict(zip(row_headers, x))) + '\n')

    f.close()


