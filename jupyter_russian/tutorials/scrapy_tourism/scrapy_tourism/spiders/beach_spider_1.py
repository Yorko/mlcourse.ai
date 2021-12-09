import scrapy

class TourismBeachSpider1(scrapy.Spider):
    name = "beach_1"
    start_urls = [
        'http://www.classification-tourism.ru/index.php/displayBeach/index',
    ]

    def parse(self, response):
        for obj in response.css('a.field.object-title'):
            yield {
                'title': obj.css('::text').extract_first()
            }

        next_page_href = response.css('li.next a::attr("href")').extract_first()
        if next_page_href is not None:
            next_page_url = response.urljoin(next_page_href)
            yield scrapy.Request(next_page_url, self.parse)

