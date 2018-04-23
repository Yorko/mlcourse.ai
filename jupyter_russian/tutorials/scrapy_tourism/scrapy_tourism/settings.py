BOT_NAME = 'scrapy_tourism'

SPIDER_MODULES = ['scrapy_tourism.spiders']
NEWSPIDER_MODULE = 'scrapy_tourism.spiders'

ROBOTSTXT_OBEY = True

DOWNLOAD_DELAY = 0.25

# ITEM_PIPELINES = {
#    'scrapy_tourism.pipelines.GeocoderPipeline': 300,
# }

