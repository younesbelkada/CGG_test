import torch
from sklearn import metrics
import numpy as np

def acc(target, y_batch):
	m = metrics.confusion_matrix(y_batch.view(-1).cpu(), target.view(-1).cpu())
	if len(m) == 1:
		return 100, 0, 1, 0
	else:
		acc0 = m[0,0]*100/(m[0,0]+m[0,1])
		if m[1,0]+m[1,1] != 0:
			acc1 = m[1,1]*100/(m[1,0]+m[1,1])
		else:
			acc1 = 0.0
		iou0 = m[0,0]/(m[0,0]+m[0,1]+m[1,0])
		iou1 = m[1,1]/(m[1,1]+m[0,1]+m[1,0])
		return acc0, acc1, iou0, iou1

def acc_multi(target, y_batch):
	return target.eq(y_batch).float().mean()

