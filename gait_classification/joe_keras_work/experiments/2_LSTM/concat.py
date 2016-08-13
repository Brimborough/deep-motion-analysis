import os

s = "concat:"
for i in [1,2,5,8,10]:
	s+= "lstm256512s"+str(i)+".mp4|"
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
print(os.getcwd() + "\n")
print(s[:-1] + "\n")
os.system('ffmpeg -i "'+str(s[:-1])+'" -codec copy output.mp4')