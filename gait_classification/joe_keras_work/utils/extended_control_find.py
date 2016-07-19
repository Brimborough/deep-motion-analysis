import numpy as np
import pprint

def extended_control():
	control = np.load('../../data/Joe/edin_shuffled_control.npz')['control']
	control = control.swapaxes(1,2)
	train_control = control[:310,8::8]
	test_control = control[310:,8::8]
	# Test, training and then time
	sames = {}
	for x in range(0,310):
            for i in [0]:
            	for j in [15]:
	            	for y in [0,1,2,3,5,7,8,9,10]:
		            	if (np.array_equal(test_control[y,j], train_control[x,i])):
		            		print x,i,y,j
		            		if not (sames.has_key(str(y))):
		            			sames.update({str(y): [str(j),str(x), str(i)]})

	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(sames)
	
	sames2 = {}
	for x in range(0,310):
            for i in [0,14,15,16]:
            	for j in [14,15,16]:
	            	for y in [4,6]:
		            	if (np.array_equal(test_control[y,j], train_control[x,i])):
		            		print x,i,y,j
		            		if not (sames2.has_key(str(y))):
		            			sames2.update({str(y): [str(j),str(x), str(i)]})

	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(sames2)

extended_control()