import time
from datetime import datetime as dt
while True:
    if dt.now().hour in [2]:
        print "yes"
        time.sleep(3600)
    else:
        print "no"
        time.sleep(20)
def test():
    open('myfile','w')
    f.write('hi there\n') # python will convert \n to os.linesep
    f.close() 
