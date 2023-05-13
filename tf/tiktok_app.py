# from selenium import webdriver
# import time
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys



# options = webdriver.EdgeOptions()
# options.add_experimental_option("detach", True)
# options.add_experimental_option("excludeSwitches", ['enable-automation'])
# ff = webdriver.Edge(options=options)
# ff.get("https://www.tiktok.com/")
# ff.find_element(By.XPATH, "//div[contains(@class, 'DivVideoPlayerContainer')]").click()


# while True:
#     time.sleep(3)
#     ff.find_element(By.TAG_NAME, 'body').send_keys(Keys.DOWN)

import tensorflow as tf
import pandas as pd
import numpy as np

up_df = pd.read_csv("RIGHTHANDDOWN.txt")
X = []
Y = []
no_of_timesteps = 27


dataset = up_df.iloc[:, 1:].values
n_sample = len(dataset)

for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps : i, :])
    Y.append(1)

X, Y = np.array(X), np.array(Y)

model = tf.keras.models.load_model("model_hand2.h5")

print(X[0:27].shape)
results = model.predict(X[0 : 27])
print(results)


