import http.client
import urllib.parse
import json

# **********************************************
# *** Update or verify the following values. ***
# **********************************************

# Replace the subscriptionKey string value with your valid subscription key.
subscriptionKey = '1f5276a0d951474cb672826b30098ed2'

host = 'api.cognitive.microsoft.com'
path = '/bing/v7.0/entities'


def get_suggestions(query):
    headers = {'Ocp-Apim-Subscription-Key': subscriptionKey}
    conn = http.client.HTTPSConnection(host)

    mkt = 'en-US'
    params = '?mkt=' + mkt + '&q=' + urllib.parse.quote(query)

    conn.request("GET", path + params, None, headers)
    response = conn.getresponse()
    return response.read()

# result = get_suggestions('colin kaepernick')
# print (json.dumps(json.loads(result), indent=4))