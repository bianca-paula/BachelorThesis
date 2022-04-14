import requests
from summa import summarizer


# location given here
location = "ibuprofen"
URL = "https://api.fda.gov/drug/label.json"
# defining a params dict for the parameters to be sent to the API
PARAMS = {'search':'contraindications:{}'.format(location)}

# sending get request and saving the response as response object
r = requests.get(url=URL, params=PARAMS)

# extracting data in json format
data = r.json()

# extracting latitude, longitude and formatted address
# of the first matching location
latitude = data['results'][0]['precautions'][0].replace(". ", ".\n")
print(latitude)

print("**************************************************************************************8")


# printing the output
print(summarizer.summarize(latitude))
