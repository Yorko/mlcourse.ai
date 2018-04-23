import urllib.request
import requests

class GeocoderPipeline(object):
    def process_item(self, item, spider):
        address = item['address']
        print(address)
        geocode_url = 'https://geocode-maps.yandex.ru/1.x/?format=json&geocode={0}'.format(
            urllib.request.quote(address))
        response = requests.get(geocode_url)

        lat, lon = None, None
        if response.status_code == 200:
            data = response.json()
            geocoder_objects = data['response']['GeoObjectCollection']['featureMember']
            if geocoder_objects:
                coordinates = geocoder_objects[0]['GeoObject']['Point']['pos'].split()
                lat, lon = float(coordinates[1]), float(coordinates[0])

        item['lat'] = lat
        item['lon'] = lon

        return item

