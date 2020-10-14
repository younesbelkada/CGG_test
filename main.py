from data import *
from network import *
from utils import *
import argparse

input_dir = './seep_detection/train_images_256'
mask_dir = './seep_detection/train_masks_256'


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(help='Different parsers for main actions', dest='command')
predict_parser = subparsers.add_parser("predict")
training_parser = subparsers.add_parser("train")

training_parser.add_argument('--bs', type=int, default=16, help='input batch size')
training_parser.add_argument('--lr', type=float, default=0.0001, help='input batch size')
training_parser.add_argument('--epochs', type=int, default=200, help='input batch size')

training_parser.add_argument('--input_dir', type=str, default=input_dir, help='input dir')
training_parser.add_argument('--mask_dir', type=str, default=mask_dir, help='mask dir')
training_parser.add_argument('--multi_class', type=bool, default=False, help='input batch size')

args = parser.parse_args()

if args.command == 'train':
	if args.multi_class==True:
		Dataset = SeepDataset(args.input_dir, args.mask_dir, True)
		model = UNET(8).cuda()
	else:
		Dataset = SeepDataset(args.input_dir, args.mask_dir, False)
		model = UNET(2).cuda()
	train_loader = DataLoader(dataset=Dataset, batch_size=args.bs, shuffle=True)
	loss_func = nn.CrossEntropyLoss().cuda()
	opt = torch.optim.Adam(model.parameters(), lr=args.lr)
	acc1 = []
	acc0 = []
	iou0 = []
	iou1 = []
	for i in range(args.epochs):
		for x_batch, y_batch in train_loader:
			opt.zero_grad()
			x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
			output = model(x_batch)
			target = torch.argmax(output, 1)
			loss = loss_func(output.squeeze(), y_batch)
			loss.backward()
			model.eval()
			if args.multi_class == False:
				a0, a1, i0, i1 = acc(target, y_batch)
				acc1.append(a1)
				acc0.append(a0)
				iou0.append(i0)
				iou1.append(i1)
			model.train()
			opt.step()
		model.eval()
		loss = loss_func(model(x_batch).squeeze(), y_batch)
		if args.multi_class == False:
			print('epoch {} | loss : {} | acc0 : {} | acc1 : {} | iou0 : {} | iou1 : {}'.format(i+1, loss.item(), np.mean(acc0), np.mean(acc1), np.mean(iou0), np.mean(iou1)))
		else:
			accs = acc_multi(target, y_batch)
			print('epoch {} | loss : {} | mean_acc : {} '.format(i+1, loss.item(), accs*100))
		model.train()
