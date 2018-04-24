import scrapy

class TourismBeachSpider2(scrapy.Spider):
    name = "beach_2"
    start_urls = [
        'http://www.classification-tourism.ru/index.php/displayBeach/index',
    ]

    def parse_item(self, response):
        fields = {}
        for obj in response.css('div.detail-field'):
            field_name = obj.css('span.detail-label::text').extract_first()
            field_value = obj.css('span.detail-value::text').extract_first()
            fields[field_name] = field_value

        yield {
            'reg_id': fields['Регистрационный номер в Федеральном перечне:'],
            'full_name': fields['Полное наименование классифицированного объекта:'],
            'name': fields['Cокращенное наименование классифицированного объекта:'],
            'category': fields['Присвоенная категория:'],
            'address': fields['Адрес:'],
        }


    def parse(self, response):
        for obj in response.css('a.field.object-title'):
            item_href = obj.css('::attr("href")').extract_first()
            yield scrapy.Request(response.urljoin(item_href), self.parse_item)

        next_page_href = response.css('li.next a::attr("href")').extract_first()
        if next_page_href is not None:
            next_page_url = response.urljoin(next_page_href)
            yield scrapy.Request(next_page_url, self.parse)

