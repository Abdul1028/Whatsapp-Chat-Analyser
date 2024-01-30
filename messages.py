from instabot import Bot
bot = Bot()
bot.login(username="abdul_s_h_k2", password="")

# ######  upload a picture #######
# bot.upload_photo("yoda.jpg", caption="biscuit eating baby")

######  follow someone #######
bot.follow("elonrmuskk")

######  send a message #######
bot.send_message("Hello from Abdul", ['abduldotexe','shaikhabdulrasool'])
#
# ######  get follower info #######
# my_followers = bot.get_user_followers("dhavalsays")
# for follower in my_followers:
#     print(follower)
#
# bot.unfollow_everyone()