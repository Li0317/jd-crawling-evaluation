import requests
import json
import random

headers = {
	'Accept':'*/*',
	'Accept-Encoding': 'gzip, deflate, br',
	'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
	'Connection': 'keep-alive',
	'Cookie': '__jdu=16017040729121979421462; shshshfpb=umAZiNwyfj73STLjre5wlsw%3D%3D; shshshfpa=1d34c41a-75d9-5bca-05a5-d68b5e072960-1596163836; TrackID=1WaZ6GSgZdQOOUGNcKSTkiVwtzjforkoe9rYGmb4X4zZCZQMNLqFFeKxW8dFPzHvk3Elf7m7b0Lrt_aKrsFv-ZOj2WgvL_1a8GabNlxe9-PU; areaId=17; _pst=-%E6%9C%A8%E6%9E%9D; unick=-%E6%9C%A8%E6%9E%9D; pin=-%E6%9C%A8%E6%9E%9D; _tp=N%2BgYbWKJinbP2IX6PIoJqvDvqGHAXm2OiB5Si2B2caw%3D; ipLoc-djd=17-2983-23654-0; jwotest_product=99; thor=96EE95FDAA28B3A774911C244F19371D94909F3618E64D5AE2CB922F9FE91F08EF45362145DF15A96873DD4EB48C28F543E50212F501E96F82FC3405BF0BC05C6D53AF8B2528746A9C260A39E4403D0CF3BD2AC617A37857D4A19AECD65C4A57D29427D7075DAC11D1F14CB66E60205D2FF5E41AEEC2C0389A35284469AC40DA; pinId=Nmkens0Tl_Q; __jda=122270672.16017040729121979421462.1601704073.1611647965.1611915334.40; __jdc=122270672; 3AB9D23F7A4B3C9B=V7JPLRBVS3X2YET2BAAZ7C2IHYDEGU53DXMQYHARVWFEJNJQO2DUUJCCSJVCENQYF4HXZPAD2R5DAL35MLVHMTGBAY; unpl=V2_ZzNtbUNUFEB2DE5dKUsMAGJQF1xKUBdHIA8WB3JLDwRkAhsOclRCFnUUR1RnGlQUZwoZXUVcQhBFCEdkexhdBGUHFF1EVXMlRQtGZHopXAJmAxRcR1ZFFn0IRVRzH1QAZAMXWkFncxV9DHZUfRxVAWIDIgQSCgVRRQBOXXMcWg1hBSJcclZzQxsIR1V6GFoGYwASEEJQQhVzCUNVfRpUBWQDGltKUkAVcA9FZHopXw%3d%3d; CCC_SE=ADC_wgxUdaLYBMrinTBUmaolYrVSDqyoNYE9Oyi%2fbKrsZgt6pEvvpQ%2fe6htaqDI8972Y3n80trQyERgv3Vr2wxV%2f43IJj7YHCbdy44usWeg7tb0OsRpOaILdc6aLbF4SerppZcQzmx5LdDc52m2xqD27t%2bBLU6tShKEfm3sVkDLEiLNFygwZ3zGYPrHg%2bq217WO8jgIygU9u5YiPkSQwzovyg4dtuWPT%2bX%2fiXuPuFyTWSihbDUI5ET04WDU1JL9UKXBvQc9pMytVujYUzGFr53x%2b7GqRdSr5IIvfLAJPPujvl9RubGlmLaz%2bbIgrhbS2iG1YsfwrL4f4FFoTGGos%2fkkwm23Y0KztidFfCRbbCA79rhebIeeEQlXmW4wMC8DpWolYR9VXW6j4nUqke6%2fd1DdGxSbItaNVq8BRdWz%2b1CcGDeBlzzG%2fkGcfR%2bFGf41piqpVpoiCbXrD5DJkxr8BYDl%2b%2bxX3Y56DeR0CpfQT6Bb%2fB4H1OiOk3Ic8V9bGXpEhIhVpN2sx4OU7Q5zo0gggCaVJeg%3d%3d; shshshfp=c92453e15b5ade0ecf4331aa6617dc2b; shshshsID=82894d30d0e8e77f24e05ae56d9d86a7_4_1611915373011; __jdb=122270672.8.16017040729121979421462|40.1611915334; __jdv=122270672|kong|t_1000357173_|tuiguang|02fd2598cca44b4096ecd6ab8cb0208b|1611915373039',
	'Host': 'club.jd.com',
	'Referer': 'https://item.jd.com/',
	'sec-ch-ua': '"Chromium";v="88", "Google Chrome";v="88", ";Not A Brand";v="99"',
	'sec-ch-ua-mobile': '?0',
	'Sec-Fetch-Dest': 'script',
	'Sec-Fetch-Mode': 'no-cors',
	'Sec-Fetch-Site': 'same-site',
	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36'
}			#设置headers
content = []  #存放评价

try:
	print("正在执行请稍后，waitting")
	def crawler(url):
		req=requests.get(url ,timeout=random.uniform(30, 50),headers=headers)  # 获取网页信息
		jd=json.loads(req.text.lstrip("fetchJSON_comme nt98vv375(").rstrip(");"))	#json解析
		for i in jd['comments']:
			content.append(i['content'])
			content.append(i['creationTime'])
			content.append(i['nickname'])
		return content
	f = open("评价.txt",'a')
	for i in range(50,100):												#抓取页面数
		url="https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=1748541&score=0&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1".format(i)
		crawler(url)
		for line in content:
			f.write(line)
			f.write('\n')

except Exception as e:
	print(e)
	print("暂时被封禁，请稍后再试")
	print("已获取部分评价在txt文档")

finally:
	f.close()