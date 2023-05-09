from selenium import webdriver
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys



options = webdriver.EdgeOptions()
options.add_experimental_option("detach", True)
options.add_experimental_option("excludeSwitches", ['enable-automation'])
ff = webdriver.Edge(options=options)
ff.get("https://www.tiktok.com/")
ff.find_element(By.XPATH, "//div[contains(@class, 'DivVideoPlayerContainer')]").click()


while True:
    time.sleep(3)
    ff.find_element(By.TAG_NAME, 'body').send_keys(Keys.DOWN)
